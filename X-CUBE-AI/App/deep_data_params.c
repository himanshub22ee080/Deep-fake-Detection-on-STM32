/**
  ******************************************************************************
  * @file    deep_data_params.c
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

#include "deep_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_deep_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_deep_weights_array_u64[230] = {
  0xf7923ba68c811eb4U, 0xa33681819f31894bU, 0x36829a8fc591f5fbU, 0xd5474eU,
  0x0U, 0x2744a081134bb581U, 0x81a8d1817faa047fU, 0x4b7fb7ba81b8696dU,
  0xfffffffffffffff9U, 0x7U, 0xfffffffc00000007U, 0xfffffffcffffffffU,
  0x2f162381b7244881U, 0x2e17cdb4e0f320a6U, 0xfc99b99fab3283fdU, 0xf27fb6f3815764acU,
  0x811e812de7ec5547U, 0x305d971847ff002U, 0xbe2a22a97477df8U, 0x92e3aa04c9e37f25U,
  0xa5ce897e872d4107U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x7fe726d13309d2d5U, 0x643f350a815350d0U, 0x3ea1bfb87f25d196U,
  0x7ad6304ef2b77fffU, 0x18d1a5e42ea14f81U, 0xb0303811920dacbU, 0x290a3381cbe236c9U,
  0x42a7352681caee31U, 0xcfd3c47d7f778503U, 0xea1e8138f522b4a6U, 0xc881ad7c05a3fa63U,
  0x1f81de461e241730U, 0xbb07cd197febd312U, 0xcbb3c98134d1fbc8U, 0x12297fad73ed47f8U,
  0xfc7ffa724159e033U, 0xfffffffeU, 0x200000002U, 0x2U,
  0x0U, 0xfffffffdU, 0xffffffffffffffffU, 0x1fffffffeU,
  0xfffffffe00000001U, 0xfcecce7f57952eb2U, 0xdc9ecc22fce6b9d5U, 0xeff8973b2cc10b96U,
  0xd5edee04f17fa281U, 0xec02817a16812281U, 0xe6f2073e1863d7a6U, 0xcff76125d2bc4019U,
  0xfa8901be5ecc59f8U, 0xa04028ebe71f090cU, 0xdec18b81fa61e9a8U, 0x9643fe53dfd644f8U,
  0xdde0f0b27f0b7fd2U, 0xd649b5f2b1c17810U, 0x1815ee84cad09c1U, 0x9330be9d811a7af7U,
  0x7fbf688db141cb81U, 0x817fd23b9cbb7ffdU, 0x77107fa2eb8d50e0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0xb4e210fedfc4e481U,
  0x9a6c26233128fdc4U, 0xf21cacf9c07f50a3U, 0x852b07456e21a04U, 0x4ed0780cf1c5d2d6U,
  0xf30703078104b6eeU, 0xa28cdd387ff3fbcdU, 0x6812ebdbe9cb9c08U, 0xe6f94a1def14f1faU,
  0xdceae802cf2081ccU, 0x1c857431a949c457U, 0x47813ce00dc97408U, 0xf5b47f106bbbb192U,
  0xfc023bc71b1946c1U, 0xf34e688ea04b5e2U, 0xd0ecf0810995ff1bU, 0x57effc1b47e1f418U,
  0x7f3219ebd9beed41U, 0x24e4c0ed35e4c97fU, 0x363e13be0fd676f9U, 0xf2a21f0a43f8be81U,
  0x5de19e2f80430e2U, 0x59fb151dffdd1201U, 0xe7ec15580aeded7fU, 0xc93a7fe5e52a0820U,
  0xecdaf7bf8af9b9a9U, 0x1a11d4c1f13b3eb1U, 0x713facf0f810b07U, 0x2f3ebdfd37dcc8U,
  0xe9e5c9c68106b6d3U, 0x12eef32a2cdbcb1bU, 0x467f1407d6b9d631U, 0x13fc7f2e22f3140eU,
  0xfc0bec06d10be50bU, 0x1a2d7f21f0232e55U, 0xbcaec19a52fcfd0U, 0xafc38db66100227fU,
  0x3a37ffb82125ffe5U, 0x468189f83012b069U, 0x521733034a077635U, 0x2bbb3317edf5ba81U,
  0xcaf60df60f1817e2U, 0x26be3df610ef81c2U, 0xf7d328eb102606feU, 0x4d0709ebe6f7e1a8U,
  0xd7cce707a51381caU, 0x260fa8d87fe2bb7cU, 0x680519a961c015f9U, 0xead239f31ce3cd81U,
  0xd1f7faf5012a35e7U, 0x7fce042d3accda25U, 0x6176391ff0d8e911U, 0xf781f8fd2cdfd350U,
  0x403a2b2436db3448U, 0x3aa5e1e0f00af80dU, 0x2d238140e2e2e3ebU, 0x46457b81df24df20U,
  0xd41682b2af2491d2U, 0x48e1dce619311e96U, 0x133dec9921810321U, 0x6f8e6a020fb8e613U,
  0xaf5f2cf381dea4f6U, 0x4151f181e750c3cdU, 0xe6eff615f21de1b4U, 0xfffffffe00000000U,
  0xfffffffe00000000U, 0x0U, 0xffffffff00000000U, 0x0U,
  0x100000000U, 0x1U, 0xffffffffU, 0x0U,
  0x1U, 0x0U, 0x0U, 0x0U,
  0x0U, 0xfffffffffffffffeU, 0xffffffff00000001U, 0x154d1298671d81adU,
  0xe81bd0621edba1e0U, 0xd9f8c48cec4e9eccU, 0x71de37756849b6a8U, 0xf4e05625cf54c9f5U,
  0xf5c6ad23240a31d6U, 0x2e31ebec37ddeb23U, 0xc37ff69a42440522U, 0x261f88cc3295f920U,
  0x6e5a1d02282fa846U, 0x9fbd394ed60bd8e9U, 0x1d813b36f7985a44U, 0x263fab0138c92219U,
  0x51694303d51dd04dU, 0xe7033014db36d3c6U, 0x25812170c2ad3539U, 0xf6c6dfe4f4b107c4U,
  0xdfe92a1aeadcb1cdU, 0x810ccdc78cf8f5fbU, 0x112af5ebd296e3dcU, 0xc21805b51f95feaeU,
  0xd22456dbb1234a29U, 0xfbd0b81dd1a2812bU, 0xe9e23fc0c920b55U, 0x5b8a311e7cb2ff4U,
  0xddfb13f4b4e2a9d6U, 0x81afc481af0307d0U, 0xebf3c1d6bfc99bb3U, 0x25f1f3141efcee1fU,
  0x1102eb051cd2812eU, 0xa7f2c1e390ff0a10U, 0xdf6ead81fede9f0U, 0xd5e20712e30201f4U,
  0x94abfc21f3eb3781U, 0x40f1e5d5312b28f4U, 0xb326e2ccf318abe7U, 0x9bc3f9dd471bc1d2U,
  0xf7e3eb22103a48baU, 0x17bc208104e488ceU, 0xba60a464e5eaf814U, 0xaf0370fce2bda19U,
  0x81cbec2416d34183U, 0x3725b9bb490e0c22U, 0xc22fef862b3da5feU, 0x92c4a17add08103dU,
  0x46e4286bee5f7baU, 0x369461bb11c59dbbU, 0xca81caeb1bb92d01U, 0xc6a3e2309370f485U,
  0x3cb74f908109f322U, 0xdbf93c62eac4fe96U, 0x95ca9d3399b62043U, 0x9e93e2deb16f207U,
  0x81a70a26dcf513b1U, 0x441bdbbf35ea2808U, 0xbf39f488014ca714U, 0x161fdcd54a810a1fU,
  0x6c48200611e0c972U, 0xa0e4e73d9adc2903U, 0x52832b5ff4ae2227U, 0xa59f3e978c2a86baU,
  0x13d5b3379b7095b3U, 0x1f91049dae33888U, 0x1d62b73ed92b0881U, 0x3fffffffeU,
  0xffffffff00000009U, 0xfffffffd00000002U, 0x4fffffffdU, 0xfffffffe00000003U,
  0xfffffffeffffffffU, 0xfffffffffffffffcU, 0xfffffffe00000000U, 0xa2cee8beb4813813U,
  0xe3d53bdc0343de5cU, 0xffffffffU,
};


ai_handle g_deep_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_deep_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

