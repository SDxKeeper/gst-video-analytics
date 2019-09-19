/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "inference_singleton.h"

#include "gva_base_inference.h"
#include "gva_utils.h"
#include "inference_impl.h"
#include <assert.h>
#include <iostream>

struct InferenceRefs {
    unsigned int numRefs = 0;
    std::list<GvaBaseInference *> elementsToInit;
    GvaBaseInference *masterElement = nullptr;
    InferenceImpl *proxy = nullptr;
};

static std::map<std::string, InferenceRefs *> inference_pool_;
static std::mutex inference_pool_mutex_;

#define COPY_GSTRING(_DST, _SRC)                                                                                       \
    g_free(_DST);                                                                                                      \
    _DST = g_strdup(_SRC);

void registerElement(GvaBaseInference *ovino, GError **error) {
    std::cout << "registerElement start"<<std::endl<<std::flush;
    try {
        std::lock_guard<std::mutex> guard(inference_pool_mutex_);
        std::string name(ovino->inference_id);
        InferenceRefs *infRefs = nullptr;

        auto it = inference_pool_.find(name);
        if (it == inference_pool_.end()) {
            infRefs = new InferenceRefs;
            ++infRefs->numRefs;
            infRefs->proxy = nullptr;
            if (ovino->model) {
                // save master element to indicate that this element has full properties set
                infRefs->masterElement = ovino;
            } else {
                // lazy initialization
                infRefs->elementsToInit.push_back(ovino);
            }
            inference_pool_.insert({name, infRefs});
        } else {
            infRefs = it->second;
            ++infRefs->numRefs;
            if (ovino->model) {
                // save master element to indicate that this element has full properties set
                infRefs->masterElement = ovino;
            } else {
                // lazy initialization
                infRefs->elementsToInit.push_back(ovino);
            }
        }
    } catch (const std::exception &e) {
        g_set_error(error, 1, 1, "%s", CreateNestedErrorMsg(e).c_str());
        std::cout << "registerElement catch"<<std::endl<<std::flush;
    }
    std::cout << "registerElement end"<<std::endl<<std::flush;
}

void fillElementProps(GvaBaseInference *targetElem, GvaBaseInference *masterElem, InferenceImpl *inference_impl) {
    std::cout << "fillElementProps start"<<std::endl<<std::flush;
    assert(masterElem);
    targetElem->inference = inference_impl;

    COPY_GSTRING(targetElem->model, masterElem->model);
    COPY_GSTRING(targetElem->device, masterElem->device);
    COPY_GSTRING(targetElem->model_proc, masterElem->model_proc);
    targetElem->batch_size = masterElem->batch_size;
    targetElem->every_nth_frame = masterElem->every_nth_frame;
    targetElem->nireq = masterElem->nireq;
    targetElem->cpu_streams = masterElem->cpu_streams;
    targetElem->gpu_streams = masterElem->gpu_streams;
    COPY_GSTRING(targetElem->infer_config, masterElem->infer_config);
    COPY_GSTRING(targetElem->allocator_name, masterElem->allocator_name);
    std::cout << "fillElementProps end"<<std::endl<<std::flush;
    // no need to copy inference_id because it should match already.
}

void initExistingElements(InferenceRefs *infRefs) {
    std::cout << "initExistingElements start"<<std::endl<<std::flush;
    assert(infRefs->masterElement);
    for (auto elem : infRefs->elementsToInit) {
        fillElementProps(elem, infRefs->masterElement, infRefs->proxy);
    }
    std::cout << "initExistingElements end"<<std::endl<<std::flush;
}

InferenceImpl *acquire_inference_instance(GvaBaseInference *ovino, GError **error) {
    std::cout << "acquire_inference_instance start"<<std::endl<<std::flush;
    try {
        std::lock_guard<std::mutex> guard(inference_pool_mutex_);
        std::string name(ovino->inference_id);

        InferenceRefs *infRefs = nullptr;
        auto it = inference_pool_.find(name);

        // Current ovino element with ovino->inference-id has not been registered
        assert(it != inference_pool_.end());

        infRefs = it->second;
        // if ovino is not master element, it will get all master element's properties here
        initExistingElements(infRefs);

        if (infRefs->proxy == nullptr)                 // no instance for current inference-id acquired yet
            infRefs->proxy = new InferenceImpl(ovino); // one instance for all elements with same inference-id

        std::cout << "acquire_inference_instance end"<<std::endl<<std::flush;
        return infRefs->proxy;
    } catch (const std::exception &e) {
        g_set_error(error, 1, 1, "%s", CreateNestedErrorMsg(e).c_str());
        std::cout << "acquire_inference_instance catch"<<std::endl<<std::flush;
        return nullptr;
    }
}

void release_inference_instance(GvaBaseInference *ovino) {
    std::cout << "release_inference_instance start"<<std::endl<<std::flush;
    std::lock_guard<std::mutex> guard(inference_pool_mutex_);
    std::string name(ovino->inference_id);

    auto it = inference_pool_.find(name);
    if (it == inference_pool_.end()) {
        std::cout << "release_inference_instance end1"<<std::endl<<std::flush;
        return;
    }

    InferenceRefs *infRefs = it->second;
    auto refcounter = --infRefs->numRefs;
    if (refcounter == 0) {
        delete infRefs->proxy;
        delete infRefs;
        inference_pool_.erase(name);
    }
    std::cout << "release_inference_instance end2"<<std::endl<<std::flush;
}

GstFlowReturn frame_to_classify_inference(GvaBaseInference *element, GstBuffer *buf, GstVideoInfo *info) {
    std::cout << "frame_to_classify_inference start"<<std::endl<<std::flush;
    if (!element || !element->inference) {
        GST_ERROR_OBJECT(element, "empty inference instance!!!!");
        return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    GstFlowReturn status;
    try {
        status = ((InferenceImpl *)element->inference)->TransformFrameIp(element, buf, info);
    } catch (const std::exception &e) {
        GST_ERROR_OBJECT(element, "%s", CreateNestedErrorMsg(e).c_str());
        status = GST_FLOW_ERROR;
    }
    std::cout << "frame_to_classify_inference end"<<std::endl<<std::flush;
    return status;
}

void classify_inference_sink_event(GvaBaseInference *ovino, GstEvent *event) {
    std::cout << "classify_inference_sink_event start"<<std::endl<<std::flush;
    ((InferenceImpl *)ovino->inference)->SinkEvent(event);
    std::cout << "classify_inference_sink_event end"<<std::endl<<std::flush;
}

void flush_inference_classify(GvaBaseInference *ovino) {
    std::cout << "flush_inference_classify start"<<std::endl<<std::flush;
    if (!ovino || !ovino->inference) {
        GST_ERROR_OBJECT(ovino, "empty inference instance!!!!");
        std::cout << "flush_inference_classify end1"<<std::endl<<std::flush;
        return;
    }

    ((InferenceImpl *)ovino->inference)->FlushInference();
    std::cout << "flush_inference_classify end2"<<std::endl<<std::flush;
}
