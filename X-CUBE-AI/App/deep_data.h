/**
  ******************************************************************************
  * @file    deep_data.h
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

#ifndef DEEP_DATA_H
#define DEEP_DATA_H

#include "deep_config.h"
#include "deep_data_params.h"

AI_DEPRECATED
#define AI_DEEP_DATA_ACTIVATIONS(ptr_)  \
  ai_deep_data_activations_buffer_get(AI_HANDLE_PTR(ptr_))

AI_DEPRECATED
#define AI_DEEP_DATA_WEIGHTS(ptr_)  \
  ai_deep_data_weights_buffer_get(AI_HANDLE_PTR(ptr_))


AI_API_DECLARE_BEGIN


extern const ai_u64 s_deep_weights_array_u64[230];



/*!
 * @brief Get network activations buffer initialized struct.
 * @ingroup deep_data
 * @param[in] ptr a pointer to the activations array storage area
 * @return an ai_buffer initialized struct
 */
AI_DEPRECATED
AI_API_ENTRY
ai_buffer ai_deep_data_activations_buffer_get(const ai_handle ptr);

/*!
 * @brief Get network weights buffer initialized struct.
 * @ingroup deep_data
 * @param[in] ptr a pointer to the weights array storage area
 * @return an ai_buffer initialized struct
 */
AI_DEPRECATED
AI_API_ENTRY
ai_buffer ai_deep_data_weights_buffer_get(const ai_handle ptr);

/*!
 * @brief Get network weights array pointer as a handle ptr.
 * @ingroup deep_data
 * @return a ai_handle pointer to the weights array
 */
AI_DEPRECATED
AI_API_ENTRY
ai_handle ai_deep_data_weights_get(void);


/*!
 * @brief Get network params configuration data structure.
 * @ingroup deep_data
 * @return true if a valid configuration is present, false otherwise
 */
AI_API_ENTRY
ai_bool ai_deep_data_params_get(ai_network_params* params);


AI_API_DECLARE_END

#endif /* DEEP_DATA_H */

