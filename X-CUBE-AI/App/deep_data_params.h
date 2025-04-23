/**
  ******************************************************************************
  * @file    deep_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-04-18T02:40:04+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef DEEP_DATA_PARAMS_H
#define DEEP_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_DEEP_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_deep_data_weights_params[1]))
*/

#define AI_DEEP_DATA_CONFIG               (NULL)


#define AI_DEEP_DATA_ACTIVATIONS_SIZES \
  { 74588, }
#define AI_DEEP_DATA_ACTIVATIONS_SIZE     (74588)
#define AI_DEEP_DATA_ACTIVATIONS_COUNT    (1)
#define AI_DEEP_DATA_ACTIVATION_1_SIZE    (74588)



#define AI_DEEP_DATA_WEIGHTS_SIZES \
  { 1836, }
#define AI_DEEP_DATA_WEIGHTS_SIZE         (1836)
#define AI_DEEP_DATA_WEIGHTS_COUNT        (1)
#define AI_DEEP_DATA_WEIGHT_1_SIZE        (1836)



#define AI_DEEP_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_deep_activations_table[1])

extern ai_handle g_deep_activations_table[1 + 2];



#define AI_DEEP_DATA_WEIGHTS_TABLE_GET() \
  (&g_deep_weights_table[1])

extern ai_handle g_deep_weights_table[1 + 2];


#endif    /* DEEP_DATA_PARAMS_H */
