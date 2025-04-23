/**
  ******************************************************************************
  * @file    deep.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-04-18T02:40:04+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "deep.h"
#include "deep_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_deep
 
#undef AI_DEEP_MODEL_SIGNATURE
#define AI_DEEP_MODEL_SIGNATURE     "0x8ca60c20bb15ac140b6348fcd929ba87"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-04-18T02:40:04+0530"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_DEEP_N_BATCHES
#define AI_DEEP_N_BATCHES         (1)

static ai_ptr g_deep_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_deep_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_keras_tensor_260_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 27648, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conversion_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 28813, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  pool_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 20000, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  pool_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 10816, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  pool_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  pool_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  nl_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conversion_14_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 3, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 112, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 92, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 297, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 593, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 112, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 16, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.4601149559021f),
    AI_PACK_INTQ_ZP(62)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 3,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004081656225025654f, 0.004437102470546961f, 0.005644471850246191f),
    AI_PACK_INTQ_ZP(0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.784325122833252f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0056259892880916595f, 0.007978430017828941f, 0.00407471414655447f, 0.00931591633707285f, 0.010247851721942425f, 0.005954062566161156f, 0.014734312891960144f, 0.007357026915997267f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(17.53484535217285f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.784325122833252f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005070643965154886f, 0.003438031766563654f, 0.004637734964489937f, 0.009748690761625767f, 0.004871791694313288f, 0.004112729802727699f, 0.005050037056207657f, 0.004769516177475452f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(10.515788078308105f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0070069171488285065f, 0.007106803357601166f, 0.005328614264726639f, 0.00666137645021081f, 0.0033481516875326633f, 0.009093265980482101f, 0.008023484610021114f, 0.006571033038198948f, 0.006036934442818165f, 0.008382401429116726f, 0.005748815834522247f, 0.010978516191244125f, 0.007090077735483646f, 0.009114732965826988f, 0.0064545366913080215f, 0.006445680744946003f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(49.3680419921875f),
    AI_PACK_INTQ_ZP(-25)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(10.515788078308105f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007549547124654055f, 0.007426649797707796f, 0.004938765428960323f, 0.005650689825415611f, 0.003241830738261342f, 0.00633623031899333f, 0.0075274077244102955f, 0.007870244793593884f, 0.006189507432281971f, 0.005689216312021017f, 0.0033385735005140305f, 0.006944736931473017f, 0.00606231065467f, 0.003614424029365182f, 0.00437972042709589f, 0.012416488490998745f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(18.730628967285156f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004266882780939341f, 0.004325938876718283f, 0.008495835587382317f, 0.003763834945857525f, 0.010528608225286007f, 0.005720579531043768f, 0.0064638289622962475f, 0.005000591278076172f, 0.013077853247523308f, 0.007369678001850843f, 0.00926284771412611f, 0.008180688135325909f, 0.005799777805805206f, 0.007442803122103214f, 0.008459125645458698f, 0.013272943906486034f, 0.01165242400020361f, 0.006566421128809452f, 0.005079061724245548f, 0.005824840161949396f, 0.012570326216518879f, 0.009434536099433899f, 0.00935119204223156f, 0.006812823936343193f, 0.009367261081933975f, 0.009832946583628654f, 0.008587383665144444f, 0.0061299982480704784f, 0.005628003738820553f, 0.0061762467958033085f, 0.006721557583659887f, 0.006809422746300697f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_UINTQ_ZP(0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.414129734039307f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0026120622642338276f, 0.008299561217427254f, 0.007936407811939716f, 0.008002621121704578f, 0.010059471242129803f, 0.0034197389613837004f, 0.006598024163395166f, 0.015755649656057358f, 0.010434230789542198f, 0.0031740590929985046f, 0.009338468313217163f, 0.0027096602134406567f, 0.0027157124131917953f, 0.009679349139332771f, 0.007168521638959646f, 0.0029926649294793606f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_12_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.4516019821167f),
    AI_PACK_INTQ_ZP(42)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_12_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012006716802716255f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.439577579498291f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_3_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.784325122833252f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(10.515788078308105f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_9_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(18.730628967285156f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_keras_tensor_260_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0f),
    AI_PACK_UINTQ_ZP(0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 96, 96), AI_STRIDE_INIT(4, 1, 1, 3, 288),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_pad_before_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 98, 98), AI_STRIDE_INIT(4, 1, 1, 3, 294),
  1, &conv2d_1_pad_before_output_array, &conv2d_1_pad_before_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 1, 1), AI_STRIDE_INIT(4, 1, 1, 112, 112),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 3, 3, 3, 1), AI_STRIDE_INIT(4, 1, 3, 3, 9),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 96, 96), AI_STRIDE_INIT(4, 1, 1, 8, 768),
  1, &conv2d_2_output_array, &conv2d_2_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 92, 1, 1), AI_STRIDE_INIT(4, 1, 1, 92, 92),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 3, 1, 1, 8), AI_STRIDE_INIT(4, 1, 3, 24, 24),
  1, &conv2d_2_weights_array, &conv2d_2_weights_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 48, 48), AI_STRIDE_INIT(4, 1, 1, 8, 384),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_pad_before_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 50, 50), AI_STRIDE_INIT(4, 1, 1, 8, 400),
  1, &conv2d_4_pad_before_output_array, &conv2d_4_pad_before_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 297, 1, 1), AI_STRIDE_INIT(4, 1, 1, 297, 297),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 1), AI_STRIDE_INIT(4, 1, 8, 8, 24),
  1, &conv2d_4_weights_array, &conv2d_4_weights_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_5_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 48, 48), AI_STRIDE_INIT(4, 1, 1, 16, 768),
  1, &conv2d_5_output_array, &conv2d_5_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 1, 1, 192, 192),
  1, &conv2d_5_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_weights, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 8, 1, 1, 16), AI_STRIDE_INIT(4, 1, 8, 128, 128),
  1, &conv2d_5_weights_array, &conv2d_5_weights_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 24, 24), AI_STRIDE_INIT(4, 1, 1, 16, 384),
  1, &conv2d_7_output_array, &conv2d_7_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_pad_before_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 26, 26), AI_STRIDE_INIT(4, 1, 1, 16, 416),
  1, &conv2d_7_pad_before_output_array, &conv2d_7_pad_before_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 593, 1, 1), AI_STRIDE_INIT(4, 1, 1, 593, 593),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 1), AI_STRIDE_INIT(4, 1, 16, 16, 48),
  1, &conv2d_7_weights_array, &conv2d_7_weights_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_8_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 24, 24), AI_STRIDE_INIT(4, 1, 1, 32, 768),
  1, &conv2d_8_output_array, &conv2d_8_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 1, 1, 384, 384),
  1, &conv2d_8_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 1, 16, 512, 512),
  1, &conv2d_8_weights_array, &conv2d_8_weights_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conversion_0_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 96, 96), AI_STRIDE_INIT(4, 1, 1, 3, 288),
  1, &conversion_0_output_array, &conversion_0_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conversion_14_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &conversion_14_output_array, &conversion_14_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_11_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_11_output_array, &gemm_11_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_scratch0, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 1, 1), AI_STRIDE_INIT(4, 2, 2, 224, 224),
  1, &gemm_11_scratch0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_weights, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 32, 16, 1, 1), AI_STRIDE_INIT(4, 1, 32, 512, 512),
  1, &gemm_11_weights_array, &gemm_11_weights_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_bias, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_12_bias_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_12_output_array, &gemm_12_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_scratch0, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 2, 2, 32, 32),
  1, &gemm_12_scratch0_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_weights, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 16, 1, 1, 1), AI_STRIDE_INIT(4, 1, 16, 16, 16),
  1, &gemm_12_weights_array, &gemm_12_weights_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  nl_13_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_13_output_array, &nl_13_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  pool_10_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &pool_10_output_array, &pool_10_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  pool_3_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 48, 48), AI_STRIDE_INIT(4, 1, 1, 8, 384),
  1, &pool_3_output_array, &pool_3_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  pool_6_output, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 24, 24), AI_STRIDE_INIT(4, 1, 1, 16, 384),
  1, &pool_6_output_array, &pool_6_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  pool_9_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 12, 12), AI_STRIDE_INIT(4, 1, 1, 32, 384),
  1, &pool_9_output_array, &pool_9_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_keras_tensor_260_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 96, 96), AI_STRIDE_INIT(4, 1, 1, 3, 288),
  1, &serving_default_keras_tensor_260_output_array, &serving_default_keras_tensor_260_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_14_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_14_chain,
  NULL, &conversion_14_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_13_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 0, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_13_nl_params, AI_ARRAY_FORMAT_S8,
    nl_13_nl_params_data, nl_13_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_13_layer, 13,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_13_chain,
  NULL, &conversion_14_layer, AI_STATIC, 
  .nl_params = &nl_13_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_12_weights, &gemm_12_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_12_layer, 12,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &gemm_12_chain,
  NULL, &nl_13_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_11_weights, &gemm_11_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_11_layer, 11,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_11_chain,
  NULL, &gemm_12_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_10_layer, 10,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_10_chain,
  NULL, &gemm_11_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(12, 12), 
  .pool_stride = AI_SHAPE_2D_INIT(12, 12), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_9_layer, 9,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_9_chain,
  NULL, &pool_10_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &conv2d_8_chain,
  NULL, &pool_9_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_3x3_sssa8_ch,
  &conv2d_7_chain,
  NULL, &conv2d_8_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_7_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_7_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_7_pad_before_value_data, conv2d_7_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_pad_before_layer, 7,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_7_pad_before_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .value = &conv2d_7_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_6_layer, 6,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_6_chain,
  NULL, &conv2d_7_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_5_weights, &conv2d_5_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_5_layer, 5,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &conv2d_5_chain,
  NULL, &pool_6_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_3x3_sssa8_ch,
  &conv2d_4_chain,
  NULL, &conv2d_5_layer, AI_STATIC, 
  .groups = 8, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_4_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_4_pad_before_value_data, conv2d_4_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_pad_before_layer, 4,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_4_pad_before_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .value = &conv2d_4_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_3_layer, 3,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_3_chain,
  NULL, &conv2d_4_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &conv2d_2_chain,
  NULL, &pool_3_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_3x3_sssa8_ch,
  &conv2d_1_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .groups = 3, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_1_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_1_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_1_pad_before_value_data, conv2d_1_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_pad_before_layer, 1,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_1_pad_before_chain,
  NULL, &conv2d_1_layer, AI_STATIC, 
  .value = &conv2d_1_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_keras_tensor_260_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_0_layer, 0,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_0_chain,
  NULL, &conv2d_1_pad_before_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1836, 1, 1),
    1836, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 74588, 1, 1),
    74588, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DEEP_IN_NUM, &serving_default_keras_tensor_260_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DEEP_OUT_NUM, &conversion_14_output),
  &conversion_0_layer, 0x7c06f335, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1836, 1, 1),
      1836, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 74588, 1, 1),
      74588, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DEEP_IN_NUM, &serving_default_keras_tensor_260_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DEEP_OUT_NUM, &conversion_14_output),
  &conversion_0_layer, 0x7c06f335, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool deep_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_deep_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_keras_tensor_260_output_array.data = AI_PTR(g_deep_activations_map[0] + 46736);
    serving_default_keras_tensor_260_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 46736);
    conversion_0_output_array.data = AI_PTR(g_deep_activations_map[0] + 46736);
    conversion_0_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 46736);
    conv2d_1_pad_before_output_array.data = AI_PTR(g_deep_activations_map[0] + 17920);
    conv2d_1_pad_before_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 17920);
    conv2d_1_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 46736);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 46736);
    conv2d_1_output_array.data = AI_PTR(g_deep_activations_map[0] + 46848);
    conv2d_1_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 46848);
    conv2d_2_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 74496);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 74496);
    conv2d_2_output_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_2_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    pool_3_output_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    pool_3_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_4_pad_before_output_array.data = AI_PTR(g_deep_activations_map[0] + 18432);
    conv2d_4_pad_before_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 18432);
    conv2d_4_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_4_output_array.data = AI_PTR(g_deep_activations_map[0] + 38432);
    conv2d_4_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 38432);
    conv2d_5_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_5_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_5_output_array.data = AI_PTR(g_deep_activations_map[0] + 192);
    conv2d_5_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 192);
    pool_6_output_array.data = AI_PTR(g_deep_activations_map[0] + 37056);
    pool_6_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 37056);
    conv2d_7_pad_before_output_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_7_pad_before_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_7_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 10816);
    conv2d_7_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 10816);
    conv2d_7_output_array.data = AI_PTR(g_deep_activations_map[0] + 11412);
    conv2d_7_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 11412);
    conv2d_8_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_8_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    conv2d_8_output_array.data = AI_PTR(g_deep_activations_map[0] + 20628);
    conv2d_8_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 20628);
    pool_9_output_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    pool_9_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    pool_10_output_array.data = AI_PTR(g_deep_activations_map[0] + 4608);
    pool_10_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 4608);
    gemm_11_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    gemm_11_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    gemm_11_output_array.data = AI_PTR(g_deep_activations_map[0] + 224);
    gemm_11_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 224);
    gemm_12_scratch0_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    gemm_12_scratch0_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    gemm_12_output_array.data = AI_PTR(g_deep_activations_map[0] + 32);
    gemm_12_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 32);
    nl_13_output_array.data = AI_PTR(g_deep_activations_map[0] + 0);
    nl_13_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 0);
    conversion_14_output_array.data = AI_PTR(g_deep_activations_map[0] + 4);
    conversion_14_output_array.data_start = AI_PTR(g_deep_activations_map[0] + 4);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool deep_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_deep_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_deep_weights_map[0] + 0);
    conv2d_1_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 0);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_deep_weights_map[0] + 28);
    conv2d_1_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 28);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_deep_weights_map[0] + 40);
    conv2d_2_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 40);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_deep_weights_map[0] + 64);
    conv2d_2_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 64);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_deep_weights_map[0] + 96);
    conv2d_4_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 96);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_deep_weights_map[0] + 168);
    conv2d_4_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 168);
    conv2d_5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_weights_array.data = AI_PTR(g_deep_weights_map[0] + 200);
    conv2d_5_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 200);
    conv2d_5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_bias_array.data = AI_PTR(g_deep_weights_map[0] + 328);
    conv2d_5_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 328);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_deep_weights_map[0] + 392);
    conv2d_7_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 392);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_deep_weights_map[0] + 536);
    conv2d_7_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 536);
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(g_deep_weights_map[0] + 600);
    conv2d_8_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 600);
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(g_deep_weights_map[0] + 1112);
    conv2d_8_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 1112);
    gemm_11_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_11_weights_array.data = AI_PTR(g_deep_weights_map[0] + 1240);
    gemm_11_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 1240);
    gemm_11_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_11_bias_array.data = AI_PTR(g_deep_weights_map[0] + 1752);
    gemm_11_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 1752);
    gemm_12_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_12_weights_array.data = AI_PTR(g_deep_weights_map[0] + 1816);
    gemm_12_weights_array.data_start = AI_PTR(g_deep_weights_map[0] + 1816);
    gemm_12_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_12_bias_array.data = AI_PTR(g_deep_weights_map[0] + 1832);
    gemm_12_bias_array.data_start = AI_PTR(g_deep_weights_map[0] + 1832);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_deep_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_DEEP_MODEL_NAME,
      .model_signature   = AI_DEEP_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1498231,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7c06f335,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_deep_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_DEEP_MODEL_NAME,
      .model_signature   = AI_DEEP_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1498231,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7c06f335,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_deep_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_deep_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_deep_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_deep_create(network, AI_DEEP_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_deep_data_params_get(&params) != true) {
    err = ai_deep_get_error(*network);
    return err;
  }
#if defined(AI_DEEP_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_DEEP_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_deep_init(*network, &params) != true) {
    err = ai_deep_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_deep_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_deep_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_deep_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_deep_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= deep_configure_weights(net_ctx, params);
  ok &= deep_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_deep_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_deep_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_DEEP_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

