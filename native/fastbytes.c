/*
 * Copyright 2021 ByteDance Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "native.h"

size_t lspace(const char *sp, size_t nb, size_t p) {
    const char * ss = sp;

    /* seek to `p` */
    sp += p;
    nb -= p;

    /* likely to run into non-spaces within a few characters, try scalar code first */
#if USE_AVX2
    __m256i space_tab = _mm256_setr_epi8(
        '\x20', 0, 0, 0, 0, 0, 0, 0,
         0, '\x09', '\x0A', 0, 0, '\x0D', 0, 0,
        '\x20', 0, 0, 0, 0, 0, 0, 0,
         0, '\x09', '\x0A', 0, 0, '\x0D', 0, 0
    );

    /* 32-byte loop */
    while (likely(nb >= 32)) {
        __m256i input = _mm256_loadu_si256((__m256i*)sp);
        __m256i shuffle = _mm256_shuffle_epi8(space_tab, input);
        __m256i result = _mm256_cmpeq_epi8(input, shuffle);
        int32_t mask = _mm256_movemask_epi8(result);
        if (mask != -1) {
            return sp - ss + __builtin_ctzll(~(uint64_t)mask);
        }
        sp += 32;
        nb -= 32;
    }
#endif

    /* remaining bytes, do with scalar code */
    while (nb-- > 0) {
        switch (*sp++) {
            case ' '  : break;
            case '\r' : break;
            case '\n' : break;
            case '\t' : break;
            default   : return sp - ss - 1;
        }
    }

    /* all the characters are spaces */
    return sp - ss;
}

#ifdef _MSC_VER
#include <xmmintrin.h>
#include <mmintrin.h>
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#include "mask_table.h"

size_t simd_json_compact(char* bytes, size_t len) {
    size_t pos = 0;
    __m128i spaces = _mm_set1_epi8(' ');
    __m128i newline = _mm_set1_epi8('\n');
    __m128i carriage = _mm_set1_epi8('\r');
    size_t i = 0;
    // vectorization
    for (; i + 15 < len; i += 16) {
        __m128i x = _mm_loadu_si128((const __m128i*)(bytes + i));
        __m128i xspaces = _mm_cmpeq_epi8(x, spaces);
        __m128i xnewline = _mm_cmpeq_epi8(x, newline);
        __m128i xcarriage = _mm_cmpeq_epi8(x, carriage);
        __m128i anywhite = _mm_or_si128(_mm_or_si128(xspaces, xnewline), xcarriage);
        uint64_t mask16 = _mm_movemask_epi8(anywhite);
        x = _mm_shuffle_epi8(x, *((__m128i*)despace_mask16 + (mask16 & 0x7fff)));
        _mm_storeu_si128((__m128i*)(bytes + pos), x);
        pos += 16 - _mm_popcnt_u32(mask16);
    }

    // remaining < 16 bit scalar processing
    for (; i < len; ++i) {
        char c = bytes[i];
        if (c == '\r' || c == '\n' || c == ' ')
            continue;
        bytes[pos++] = c;
    }
    return pos;
}