/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef _GST_GVA_CLASSIFY_H_
#define _GST_GVA_CLASSIFY_H_

#include <gst/base/gstbasetransform.h>

#include <gva_base_inference.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

G_BEGIN_DECLS

#define GST_TYPE_GVA_CLASSIFY (gst_gva_classify_get_type())
#define GST_GVA_CLASSIFY(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_GVA_CLASSIFY, GstGvaClassify))
#define GST_GVA_CLASSIFY_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_GVA_CLASSIFY, GstGvaClassifyClass))
#define GST_IS_GVA_CLASSIFY(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_GVA_CLASSIFY))
#define GST_IS_GVA_CLASSIFY_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_GVA_CLASSIFY))

typedef struct _GstGvaClassify {
    GvaBaseInference base_inference;
} GstGvaClassify;

typedef struct _GstGvaClassifyClass {
    GvaBaseInferenceClass base_class;
} GstGvaClassifyClass;

GType gst_gva_classify_get_type(void);

G_END_DECLS

#ifdef __cplusplus
} // extern "C"
#endif /* __cplusplus */

#endif
