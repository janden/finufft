 #include <finufft.h>
 #include <mex.h>
 #include <iostream>
 #include <cstring>
 #include <math.h>
 void copy_nufft_opts(const mxArray* om, nufft_opts *oc) {
   if(!mxIsStruct(om))
     mexErrMsgIdAndTxt("MATLAB:copy_nufft_opts:inputNotStruct","Input must be a structure.");
   mwIndex idx = 0;
   int ifield, nfields;
   const char **fname;
   nfields = mxGetNumberOfFields(om);
   fname = (const char**)mxCalloc(nfields, sizeof(*fname));
   for(ifield=0; ifield<nfields; ifield++) {
     fname[ifield] = mxGetFieldNameByNumber(om,ifield);
     if (strcmp(fname[ifield],"debug") == 0) {
       oc->debug = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"spread_debug") == 0) {
       oc->spread_debug = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"spread_sort") == 0) {
       oc->spread_sort = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"spread_kerevalmeth") == 0) {
       oc->spread_kerevalmeth = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"spread_kerpad") == 0) {
       oc->spread_kerpad = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"chkbnds") == 0) {
       oc->chkbnds = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"fftw") == 0) {
       int fftw = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
       oc->fftw = !fftw ? FFTW_ESTIMATE : FFTW_MEASURE;  
     }
     else if (strcmp(fname[ifield],"modeord") == 0) {
       oc->modeord = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"upsampfac") == 0) {
       oc->upsampfac = (FLT)*mxGetPr(mxGetFieldByNumber(om,idx,ifield));
     }
     else if (strcmp(fname[ifield],"spread_thread") == 0) {
       oc->spread_thread = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else if (strcmp(fname[ifield],"maxbatchsize") == 0) {
       oc->maxbatchsize = (int)round(*mxGetPr(mxGetFieldByNumber(om,idx,ifield)));
     }
     else
       return;
   }
   mxFree(fname);
 }
 int get_type(finufft_plan* plan) {
   return plan->type;
 }
 int get_ndim(finufft_plan* plan) {
   return plan->dim;
 }
 int64_t get_nj(finufft_plan* plan) {
   return plan->nj;
 }
 int64_t get_nk(finufft_plan* plan) {
   return plan->nk;
 }
 void get_nmodes(finufft_plan* plan, int64_t& ms, int64_t& mt, int64_t& mu) {
   ms = plan->ms ? plan->ms : 1;
   mt = plan->mt ? plan->mt : 1;
   mu = plan->mu ? plan->mu : 1;
   if(plan->dim<3) mu=1;
   if(plan->dim<2) mt=1;
 }
 int get_ntransf(finufft_plan* plan) {
   return plan->ntrans;
 }
/* --------------------------------------------------- */
/* Automatically generated by mwrap                    */
/* --------------------------------------------------- */

/* Code generated by mwrap */
/*
  Copyright statement for mwrap:

  mwrap -- MEX file generation for MATLAB and Octave
  Copyright (c) 2007-2008 David Bindel

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  You may distribute a work that contains part or all of the source code
  generated by mwrap under the terms of your choice.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include <mex.h>
#include <stdio.h>
#include <string.h>


#ifndef ulong
#  define ulong unsigned long
#endif
#ifndef uint
#  define uint  unsigned int
#endif
#ifndef uchar
#  define uchar unsigned char
#endif


/*
 * Records for call profile.
 */
int* mexprofrecord_= NULL;


/*
 * Support routines for copying data into and out of the MEX stubs
 */

void* mxWrapGetP(const mxArray* a, const char* fmt, const char** e)
{
    void* p = 0;
    mxArray* ap;
    if (mxGetClassID(a) == mxDOUBLE_CLASS && 
        mxGetM(a)*mxGetN(a) == 1 && *mxGetPr(a) == 0)
        return p;
    if (mxIsChar(a)) {
        char pbuf[128];
        mxGetString(a, pbuf, sizeof(pbuf));
        sscanf(pbuf, fmt, &p);
    } 
#ifdef R2008OO
    else if (ap = mxGetProperty(a, 0, "mwptr")) {
        return mxWrapGetP(ap, fmt, e);
    }
#endif
    if (p == 0)
        *e = "Invalid pointer";
    return p;
}

mxArray* mxWrapCreateP(void* p, const char* fmt)
{
    if (p == 0) {
        mxArray* z = mxCreateDoubleMatrix(1,1, mxREAL);
        *mxGetPr(z) = 0;
        return z;
    } else {
        char pbuf[128];
        sprintf(pbuf, fmt, p);
        return mxCreateString(pbuf);
    }
}

mxArray* mxWrapStrncpy(const char* s)
{
    if (s) {
        return mxCreateString(s);
    } else {
        mxArray* z = mxCreateDoubleMatrix(1,1, mxREAL);
        *mxGetPr(z) = 0;
        return z;
    }
}

double mxWrapGetScalar(const mxArray* a, const char** e)
{
    if (!a || mxGetClassID(a) != mxDOUBLE_CLASS || mxGetM(a)*mxGetN(a) != 1) {
        *e = "Invalid scalar argument";
        return 0;
    }
    return *mxGetPr(a);
}

char* mxWrapGetString(const mxArray* a, const char** e)
{
    char* s;
    int slen;
    if (!a || (!mxIsChar(a) && mxGetM(a)*mxGetN(a) > 0)) {
        *e = "Invalid string argument";
        return NULL;
    }
    slen = mxGetM(a)*mxGetN(a) + 1;
    s = (char*) mxMalloc(slen);
    if (mxGetM(a)*mxGetN(a) == 0)
        *s = 0;
    else
        mxGetString(a, s, slen);
    return s;
}


#define mxWrapGetArrayDef(func, T) \
T* func(const mxArray* a, const char** e)     \
{ \
    T* array; \
    int arraylen; \
    int i; \
    T* p; \
    double* q; \
    if (!a || mxGetClassID(a) != mxDOUBLE_CLASS) { \
        *e = "Invalid array argument"; \
        return 0; \
    } \
    arraylen = mxGetM(a)*mxGetN(a); \
    array = (T*) mxMalloc(mxGetM(a)*mxGetN(a) * sizeof(T)); \
    p = array; \
    q = mxGetPr(a); \
    for (i = 0; i < arraylen; ++i) \
        *p++ = (T) (*q++); \
    return array; \
}


#define mxWrapCopyDef(func, T) \
void func(mxArray* a, const T* q, int n) \
{ \
    int i; \
    double* p = mxGetPr(a); \
    for (i = 0; i < n; ++i) \
        *p++ = *q++; \
}


#define mxWrapReturnDef(func, T) \
mxArray* func(const T* q, int m, int n) \
{ \
    int i; \
    double* p; \
    if (!q) { \
        return mxCreateDoubleMatrix(0,0, mxREAL); \
    } else { \
        mxArray* a = mxCreateDoubleMatrix(m,n, mxREAL); \
        p = mxGetPr(a); \
        for (i = 0; i < m*n; ++i) \
            *p++ = *q++; \
        return a; \
    } \
}


#define mxWrapGetScalarZDef(func, T, ZT, setz) \
void func(T* z, const mxArray* a) \
{ \
    double* pr = mxGetPr(a); \
    double* pi = mxGetPi(a); \
    setz(z, (ZT) *pr, (pi ? (ZT) *pi : (ZT) 0)); \
}


#define mxWrapGetArrayZDef(func, T, ZT, setz) \
T* func(const mxArray* a, const char** e) \
{ \
    T* array; \
    int arraylen; \
    int i; \
    T* p; \
    double* qr; \
    double* qi; \
    if (!a || mxGetClassID(a) != mxDOUBLE_CLASS) { \
        *e = "Invalid array argument"; \
        return 0; \
    } \
    arraylen = mxGetM(a)*mxGetN(a); \
    array = (T*) mxMalloc(mxGetM(a)*mxGetN(a) * sizeof(T)); \
    p = array; \
    qr = mxGetPr(a); \
    qi = mxGetPi(a); \
    for (i = 0; i < arraylen; ++i) { \
        ZT val_qr = *qr++; \
        ZT val_qi = (qi ? (ZT) *qi++ : (ZT) 0); \
        setz(p, val_qr, val_qi); \
        ++p; \
    } \
    return array; \
}


#define mxWrapCopyZDef(func, T, real, imag) \
void func(mxArray* a, const T* q, int n) \
{ \
    int i; \
    double* pr = mxGetPr(a); \
    double* pi = mxGetPi(a); \
    for (i = 0; i < n; ++i) { \
        *pr++ = real(*q); \
        *pi++ = imag(*q); \
        ++q; \
    } \
}


#define mxWrapReturnZDef(func, T, real, imag) \
mxArray* func(const T* q, int m, int n) \
{ \
    int i; \
    double* pr; \
    double* pi; \
    if (!q) { \
        return mxCreateDoubleMatrix(0,0, mxCOMPLEX); \
    } else { \
        mxArray* a = mxCreateDoubleMatrix(m,n, mxCOMPLEX); \
        pr = mxGetPr(a); \
        pi = mxGetPi(a); \
        for (i = 0; i < m*n; ++i) { \
            *pr++ = real(*q); \
            *pi++ = imag(*q); \
            ++q; \
        } \
        return a; \
    } \
}

#include <complex>

typedef std::complex<double> dcomplex;
#define real_dcomplex(z) std::real(z)
#define imag_dcomplex(z) std::imag(z)
#define setz_dcomplex(z,r,i)  *z = dcomplex(r,i)

typedef std::complex<float> fcomplex;
#define real_fcomplex(z) std::real(z)
#define imag_fcomplex(z) std::imag(z)
#define setz_fcomplex(z,r,i)  *z = fcomplex(r,i)

/* Array copier definitions */
mxWrapGetArrayDef(mxWrapGetArray_bool, bool)
mxWrapCopyDef    (mxWrapCopy_bool,     bool)
mxWrapReturnDef  (mxWrapReturn_bool,   bool)
mxWrapGetArrayDef(mxWrapGetArray_char, char)
mxWrapCopyDef    (mxWrapCopy_char,     char)
mxWrapReturnDef  (mxWrapReturn_char,   char)
mxWrapGetArrayDef(mxWrapGetArray_double, double)
mxWrapCopyDef    (mxWrapCopy_double,     double)
mxWrapReturnDef  (mxWrapReturn_double,   double)
mxWrapGetArrayDef(mxWrapGetArray_float, float)
mxWrapCopyDef    (mxWrapCopy_float,     float)
mxWrapReturnDef  (mxWrapReturn_float,   float)
mxWrapGetArrayDef(mxWrapGetArray_int, int)
mxWrapCopyDef    (mxWrapCopy_int,     int)
mxWrapReturnDef  (mxWrapReturn_int,   int)
mxWrapGetArrayDef(mxWrapGetArray_int64_t, int64_t)
mxWrapCopyDef    (mxWrapCopy_int64_t,     int64_t)
mxWrapReturnDef  (mxWrapReturn_int64_t,   int64_t)
mxWrapGetArrayDef(mxWrapGetArray_long, long)
mxWrapCopyDef    (mxWrapCopy_long,     long)
mxWrapReturnDef  (mxWrapReturn_long,   long)
mxWrapGetArrayDef(mxWrapGetArray_size_t, size_t)
mxWrapCopyDef    (mxWrapCopy_size_t,     size_t)
mxWrapReturnDef  (mxWrapReturn_size_t,   size_t)
mxWrapGetArrayDef(mxWrapGetArray_uchar, uchar)
mxWrapCopyDef    (mxWrapCopy_uchar,     uchar)
mxWrapReturnDef  (mxWrapReturn_uchar,   uchar)
mxWrapGetArrayDef(mxWrapGetArray_uint, uint)
mxWrapCopyDef    (mxWrapCopy_uint,     uint)
mxWrapReturnDef  (mxWrapReturn_uint,   uint)
mxWrapGetArrayDef(mxWrapGetArray_ulong, ulong)
mxWrapCopyDef    (mxWrapCopy_ulong,     ulong)
mxWrapReturnDef  (mxWrapReturn_ulong,   ulong)
mxWrapGetScalarZDef(mxWrapGetScalar_fcomplex, fcomplex,
                    float, setz_fcomplex)
mxWrapGetArrayZDef (mxWrapGetArray_fcomplex, fcomplex,
                    float, setz_fcomplex)
mxWrapCopyZDef     (mxWrapCopy_fcomplex, fcomplex,
                    real_fcomplex, imag_fcomplex)
mxWrapReturnZDef   (mxWrapReturn_fcomplex, fcomplex,
                    real_fcomplex, imag_fcomplex)
mxWrapGetScalarZDef(mxWrapGetScalar_dcomplex, dcomplex,
                    double, setz_dcomplex)
mxWrapGetArrayZDef (mxWrapGetArray_dcomplex, dcomplex,
                    double, setz_dcomplex)
mxWrapCopyZDef     (mxWrapCopy_dcomplex, dcomplex,
                    real_dcomplex, imag_dcomplex)
mxWrapReturnZDef   (mxWrapReturn_dcomplex, dcomplex,
                    real_dcomplex, imag_dcomplex)

/* ---- nufft_plan.mw: 74 ----
 * finufft_plan* p = new();
 */
const char* stubids1_ = "o finufft_plan* = new()";

void mexStub1(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  out0_=0; /* p          */

    if (mexprofrecord_)
        mexprofrecord_[1]++;
    out0_ = new finufft_plan();
    plhs[0] = mxWrapCreateP(out0_, "finufft_plan:%p");

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 88 ----
 * nufft_opts* o = new();
 * Also at nufft_plan.mw: 94
 */
const char* stubids2_ = "o nufft_opts* = new()";

void mexStub2(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    nufft_opts*  out0_=0; /* o          */

    if (mexprofrecord_)
        mexprofrecord_[2]++;
    out0_ = new nufft_opts();
    plhs[0] = mxWrapCreateP(out0_, "nufft_opts:%p");

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 89 ----
 * finufft_default_opts(nufft_opts* o);
 * Also at nufft_plan.mw: 95
 */
const char* stubids3_ = "finufft_default_opts(i nufft_opts*)";

void mexStub3(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    nufft_opts*  in0_ =0; /* o          */

    in0_ = (nufft_opts*) mxWrapGetP(prhs[0], "nufft_opts:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[3]++;
    finufft_default_opts(in0_);

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 90 ----
 * int ier = finufft_makeplan(int type, int n_dims, int64_t[3] n_modes, int iflag, int n_transf, double tol, finufft_plan* plan, nufft_opts* o);
 * Also at nufft_plan.mw: 97
 */
const char* stubids4_ = "o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i finufft_plan*, i nufft_opts*)";

void mexStub4(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    int         in0_;    /* type       */
    int         in1_;    /* n_dims     */
    int64_t*    in2_ =0; /* n_modes    */
    int         in3_;    /* iflag      */
    int         in4_;    /* n_transf   */
    double      in5_;    /* tol        */
    finufft_plan*  in6_ =0; /* plan       */
    nufft_opts*  in7_ =0; /* o          */
    int         out0_;   /* ier        */
    int         dim8_;   /* 3          */

    dim8_ = (int) mxWrapGetScalar(prhs[8], &mw_err_txt_);

    if (mxGetM(prhs[2])*mxGetN(prhs[2]) != dim8_) {
        mw_err_txt_ = "Bad argument size: n_modes";        goto mw_err_label;
    }

    in0_ = (int) mxWrapGetScalar(prhs[0], &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    in1_ = (int) mxWrapGetScalar(prhs[1], &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mxGetM(prhs[2])*mxGetN(prhs[2]) != 0) {
        in2_ = mxWrapGetArray_int64_t(prhs[2], &mw_err_txt_);
        if (mw_err_txt_)
            goto mw_err_label;
    } else
        in2_ = NULL;
    in3_ = (int) mxWrapGetScalar(prhs[3], &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    in4_ = (int) mxWrapGetScalar(prhs[4], &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    in5_ = (double) mxWrapGetScalar(prhs[5], &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    in6_ = (finufft_plan*) mxWrapGetP(prhs[6], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    in7_ = (nufft_opts*) mxWrapGetP(prhs[7], "nufft_opts:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[4]++;
    out0_ = finufft_makeplan(in0_, in1_, in2_, in3_, in4_, in5_, in6_, in7_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;

mw_err_label:
    if (in2_)  mxFree(in2_);
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 91 ----
 * delete(nufft_opts* o);
 * Also at nufft_plan.mw: 98
 */
const char* stubids5_ = "delete(i nufft_opts*)";

void mexStub5(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    nufft_opts*  in0_ =0; /* o          */

    in0_ = (nufft_opts*) mxWrapGetP(prhs[0], "nufft_opts:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[5]++;
    delete(in0_);

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 96 ----
 * copy_nufft_opts(mxArray opts, nufft_opts* o);
 */
const char* stubids8_ = "copy_nufft_opts(i mxArray, i nufft_opts*)";

void mexStub8(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    const mxArray*  in0_;    /* opts       */
    nufft_opts*  in1_ =0; /* o          */

    in0_ = prhs[0];
    in1_ = (nufft_opts*) mxWrapGetP(prhs[1], "nufft_opts:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[8]++;
    copy_nufft_opts(in0_, in1_);

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 103 ----
 * finufft_destroy(finufft_plan* plan);
 * Also at nufft_plan.mw: 107
 */
const char* stubids11_ = "finufft_destroy(i finufft_plan*)";

void mexStub11(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[11]++;
    finufft_destroy(in0_);

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 113 ----
 * int ier = finufft_setpts(finufft_plan* plan, int64_t nj, double[] xj, double[] yj, double[] zj, int64_t nk, double[] s, double[] t, double[] u);
 */
const char* stubids13_ = "o int = finufft_setpts(i finufft_plan*, i int64_t, i double[], i double[], i double[], i int64_t, i double[], i double[], i double[])";

void mexStub13(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    int64_t     in1_;    /* nj         */
    double*     in2_ =0; /* xj         */
    double*     in3_ =0; /* yj         */
    double*     in4_ =0; /* zj         */
    int64_t     in5_;    /* nk         */
    double*     in6_ =0; /* s          */
    double*     in7_ =0; /* t          */
    double*     in8_ =0; /* u          */
    int         out0_;   /* ier        */

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    in1_ = (int64_t) mxWrapGetScalar(prhs[1], &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mxGetM(prhs[2])*mxGetN(prhs[2]) != 0) {
        in2_ = mxGetPr(prhs[2]);
    } else
        in2_ = NULL;
    if (mxGetM(prhs[3])*mxGetN(prhs[3]) != 0) {
        in3_ = mxGetPr(prhs[3]);
    } else
        in3_ = NULL;
    if (mxGetM(prhs[4])*mxGetN(prhs[4]) != 0) {
        in4_ = mxGetPr(prhs[4]);
    } else
        in4_ = NULL;
    in5_ = (int64_t) mxWrapGetScalar(prhs[5], &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mxGetM(prhs[6])*mxGetN(prhs[6]) != 0) {
        in6_ = mxGetPr(prhs[6]);
    } else
        in6_ = NULL;
    if (mxGetM(prhs[7])*mxGetN(prhs[7]) != 0) {
        in7_ = mxGetPr(prhs[7]);
    } else
        in7_ = NULL;
    if (mxGetM(prhs[8])*mxGetN(prhs[8]) != 0) {
        in8_ = mxGetPr(prhs[8]);
    } else
        in8_ = NULL;
    if (mexprofrecord_)
        mexprofrecord_[13]++;
    out0_ = finufft_setpts(in0_, in1_, in2_, in3_, in4_, in5_, in6_, in7_, in8_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 140 ----
 * int type = get_type(finufft_plan* plan);
 */
const char* stubids14_ = "o int = get_type(i finufft_plan*)";

void mexStub14(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    int         out0_;   /* type       */

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[14]++;
    out0_ = get_type(in0_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 141 ----
 * int n_transf = get_ntransf(finufft_plan* plan);
 */
const char* stubids15_ = "o int = get_ntransf(i finufft_plan*)";

void mexStub15(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    int         out0_;   /* n_transf   */

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[15]++;
    out0_ = get_ntransf(in0_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 144 ----
 * get_nmodes(finufft_plan* plan, output int64_t& ms, output int64_t& mt, output int64_t& mu);
 */
const char* stubids16_ = "get_nmodes(i finufft_plan*, o int64_t&, o int64_t&, o int64_t&)";

void mexStub16(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    int64_t     out0_;   /* ms         */
    int64_t     out1_;   /* mt         */
    int64_t     out2_;   /* mu         */

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[16]++;
    get_nmodes(in0_, out0_, out1_, out2_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[1]) = out1_;
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[2]) = out2_;

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 146 ----
 * int ier = finufft_exec(finufft_plan* plan, dcomplex[] data_in, output dcomplex[outsize] result);
 */
const char* stubids17_ = "o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[x])";

void mexStub17(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    dcomplex*   in1_ =0; /* data_in    */
    int         out0_;   /* ier        */
    dcomplex*   out1_=0; /* result     */
    int         dim2_;   /* outsize    */

    dim2_ = (int) mxWrapGetScalar(prhs[2], &mw_err_txt_);

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mxGetM(prhs[1])*mxGetN(prhs[1]) != 0) {
        in1_ = mxWrapGetArray_dcomplex(prhs[1], &mw_err_txt_);
        if (mw_err_txt_)
            goto mw_err_label;
    } else
        in1_ = NULL;
    out1_ = (dcomplex*) mxMalloc(dim2_*sizeof(dcomplex));
    if (mexprofrecord_)
        mexprofrecord_[17]++;
    out0_ = finufft_exec(in0_, in1_, out1_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;
    plhs[1] = mxCreateDoubleMatrix(dim2_, 1, mxCOMPLEX);
    mxWrapCopy_dcomplex(plhs[1], out1_, dim2_);

mw_err_label:
    if (in1_)  mxFree(in1_);
    if (out1_) mxFree(out1_);
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 149 ----
 * int64_t nj = get_nj(finufft_plan* plan);
 */
const char* stubids18_ = "o int64_t = get_nj(i finufft_plan*)";

void mexStub18(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    int64_t     out0_;   /* nj         */

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[18]++;
    out0_ = get_nj(in0_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 150 ----
 * int ier = finufft_exec(finufft_plan* plan, output dcomplex[nj, n_transf] result, dcomplex[] data_in);
 */
const char* stubids19_ = "o int = finufft_exec(i finufft_plan*, o dcomplex[xx], i dcomplex[])";

void mexStub19(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    dcomplex*   in1_ =0; /* data_in    */
    int         out0_;   /* ier        */
    dcomplex*   out1_=0; /* result     */
    int         dim2_;   /* nj         */
    int         dim3_;   /* n_transf   */

    dim2_ = (int) mxWrapGetScalar(prhs[2], &mw_err_txt_);
    dim3_ = (int) mxWrapGetScalar(prhs[3], &mw_err_txt_);

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mxGetM(prhs[1])*mxGetN(prhs[1]) != 0) {
        in1_ = mxWrapGetArray_dcomplex(prhs[1], &mw_err_txt_);
        if (mw_err_txt_)
            goto mw_err_label;
    } else
        in1_ = NULL;
    out1_ = (dcomplex*) mxMalloc(dim2_*dim3_*sizeof(dcomplex));
    if (mexprofrecord_)
        mexprofrecord_[19]++;
    out0_ = finufft_exec(in0_, out1_, in1_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;
    plhs[1] = mxCreateDoubleMatrix(dim2_, dim3_, mxCOMPLEX);
    mxWrapCopy_dcomplex(plhs[1], out1_, dim2_*dim3_);

mw_err_label:
    if (out1_) mxFree(out1_);
    if (in1_)  mxFree(in1_);
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 152 ----
 * int64_t nk = get_nk(finufft_plan* plan);
 */
const char* stubids20_ = "o int64_t = get_nk(i finufft_plan*)";

void mexStub20(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    int64_t     out0_;   /* nk         */

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mexprofrecord_)
        mexprofrecord_[20]++;
    out0_ = get_nk(in0_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;

mw_err_label:
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ---- nufft_plan.mw: 153 ----
 * int ier = finufft_exec(finufft_plan* plan, dcomplex[] data_in, output dcomplex[nk, n_transf] result);
 */
const char* stubids21_ = "o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[xx])";

void mexStub21(int nlhs, mxArray* plhs[],
              int nrhs, const mxArray* prhs[])
{
    const char* mw_err_txt_ = 0;
    finufft_plan*  in0_ =0; /* plan       */
    dcomplex*   in1_ =0; /* data_in    */
    int         out0_;   /* ier        */
    dcomplex*   out1_=0; /* result     */
    int         dim2_;   /* nk         */
    int         dim3_;   /* n_transf   */

    dim2_ = (int) mxWrapGetScalar(prhs[2], &mw_err_txt_);
    dim3_ = (int) mxWrapGetScalar(prhs[3], &mw_err_txt_);

    in0_ = (finufft_plan*) mxWrapGetP(prhs[0], "finufft_plan:%p", &mw_err_txt_);
    if (mw_err_txt_)
        goto mw_err_label;
    if (mxGetM(prhs[1])*mxGetN(prhs[1]) != 0) {
        in1_ = mxWrapGetArray_dcomplex(prhs[1], &mw_err_txt_);
        if (mw_err_txt_)
            goto mw_err_label;
    } else
        in1_ = NULL;
    out1_ = (dcomplex*) mxMalloc(dim2_*dim3_*sizeof(dcomplex));
    if (mexprofrecord_)
        mexprofrecord_[21]++;
    out0_ = finufft_exec(in0_, in1_, out1_);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(plhs[0]) = out0_;
    plhs[1] = mxCreateDoubleMatrix(dim2_, dim3_, mxCOMPLEX);
    mxWrapCopy_dcomplex(plhs[1], out1_, dim2_*dim3_);

mw_err_label:
    if (in1_)  mxFree(in1_);
    if (out1_) mxFree(out1_);
    if (mw_err_txt_)
        mexErrMsgTxt(mw_err_txt_);
}

/* ----
 */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    char id[512];
    if (nrhs == 0) {
        mexPrintf("Mex function installed\n");
        return;
    }

    if (mxGetString(prhs[0], id, sizeof(id)) != 0)
        mexErrMsgTxt("Identifier should be a string");
    else if (strcmp(id, stubids1_) == 0)
        mexStub1(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids2_) == 0)
        mexStub2(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids3_) == 0)
        mexStub3(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids4_) == 0)
        mexStub4(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids5_) == 0)
        mexStub5(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids8_) == 0)
        mexStub8(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids11_) == 0)
        mexStub11(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids13_) == 0)
        mexStub13(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids14_) == 0)
        mexStub14(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids15_) == 0)
        mexStub15(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids16_) == 0)
        mexStub16(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids17_) == 0)
        mexStub17(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids18_) == 0)
        mexStub18(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids19_) == 0)
        mexStub19(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids20_) == 0)
        mexStub20(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, stubids21_) == 0)
        mexStub21(nlhs,plhs, nrhs-1,prhs+1);
    else if (strcmp(id, "*profile on*") == 0) {
        if (!mexprofrecord_) {
            mexprofrecord_ = (int*) malloc(22 * sizeof(int));
            mexLock();
        }
        memset(mexprofrecord_, 0, 22 * sizeof(int));
    } else if (strcmp(id, "*profile off*") == 0) {
        if (mexprofrecord_) {
            free(mexprofrecord_);
            mexUnlock();
        }
        mexprofrecord_ = NULL;
    } else if (strcmp(id, "*profile report*") == 0) {
        if (!mexprofrecord_)
            mexPrintf("Profiler inactive\n");
        mexPrintf("%d calls to nufft_plan.mw:74\n", mexprofrecord_[1]);
        mexPrintf("%d calls to nufft_plan.mw:88 (nufft_plan.mw:94)\n", mexprofrecord_[2]);
        mexPrintf("%d calls to nufft_plan.mw:89 (nufft_plan.mw:95)\n", mexprofrecord_[3]);
        mexPrintf("%d calls to nufft_plan.mw:90 (nufft_plan.mw:97)\n", mexprofrecord_[4]);
        mexPrintf("%d calls to nufft_plan.mw:91 (nufft_plan.mw:98)\n", mexprofrecord_[5]);
        mexPrintf("%d calls to nufft_plan.mw:96\n", mexprofrecord_[8]);
        mexPrintf("%d calls to nufft_plan.mw:103 (nufft_plan.mw:107)\n", mexprofrecord_[11]);
        mexPrintf("%d calls to nufft_plan.mw:113\n", mexprofrecord_[13]);
        mexPrintf("%d calls to nufft_plan.mw:140\n", mexprofrecord_[14]);
        mexPrintf("%d calls to nufft_plan.mw:141\n", mexprofrecord_[15]);
        mexPrintf("%d calls to nufft_plan.mw:144\n", mexprofrecord_[16]);
        mexPrintf("%d calls to nufft_plan.mw:146\n", mexprofrecord_[17]);
        mexPrintf("%d calls to nufft_plan.mw:149\n", mexprofrecord_[18]);
        mexPrintf("%d calls to nufft_plan.mw:150\n", mexprofrecord_[19]);
        mexPrintf("%d calls to nufft_plan.mw:152\n", mexprofrecord_[20]);
        mexPrintf("%d calls to nufft_plan.mw:153\n", mexprofrecord_[21]);
    } else if (strcmp(id, "*profile log*") == 0) {
        FILE* logfp;
        if (nrhs != 2 || mxGetString(prhs[1], id, sizeof(id)) != 0)
            mexErrMsgTxt("Must have two string arguments");
        logfp = fopen(id, "w+");
        if (!logfp)
            mexErrMsgTxt("Cannot open log for output");
        if (!mexprofrecord_)
            fprintf(logfp, "Profiler inactive\n");
        fprintf(logfp, "%d calls to nufft_plan.mw:74\n", mexprofrecord_[1]);
        fprintf(logfp, "%d calls to nufft_plan.mw:88 (nufft_plan.mw:94)\n", mexprofrecord_[2]);
        fprintf(logfp, "%d calls to nufft_plan.mw:89 (nufft_plan.mw:95)\n", mexprofrecord_[3]);
        fprintf(logfp, "%d calls to nufft_plan.mw:90 (nufft_plan.mw:97)\n", mexprofrecord_[4]);
        fprintf(logfp, "%d calls to nufft_plan.mw:91 (nufft_plan.mw:98)\n", mexprofrecord_[5]);
        fprintf(logfp, "%d calls to nufft_plan.mw:96\n", mexprofrecord_[8]);
        fprintf(logfp, "%d calls to nufft_plan.mw:103 (nufft_plan.mw:107)\n", mexprofrecord_[11]);
        fprintf(logfp, "%d calls to nufft_plan.mw:113\n", mexprofrecord_[13]);
        fprintf(logfp, "%d calls to nufft_plan.mw:140\n", mexprofrecord_[14]);
        fprintf(logfp, "%d calls to nufft_plan.mw:141\n", mexprofrecord_[15]);
        fprintf(logfp, "%d calls to nufft_plan.mw:144\n", mexprofrecord_[16]);
        fprintf(logfp, "%d calls to nufft_plan.mw:146\n", mexprofrecord_[17]);
        fprintf(logfp, "%d calls to nufft_plan.mw:149\n", mexprofrecord_[18]);
        fprintf(logfp, "%d calls to nufft_plan.mw:150\n", mexprofrecord_[19]);
        fprintf(logfp, "%d calls to nufft_plan.mw:152\n", mexprofrecord_[20]);
        fprintf(logfp, "%d calls to nufft_plan.mw:153\n", mexprofrecord_[21]);
        fclose(logfp);
    } else
        mexErrMsgTxt("Unknown identifier");
}

