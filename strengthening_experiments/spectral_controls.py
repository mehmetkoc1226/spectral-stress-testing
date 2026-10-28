"""Mean/DC and AC-RMS controls for NCHW arrays; NumPy or PyTorch.

The pixel intervention is signed and is never clipped. Per-channel means and
population AC-RMS are matched before checkpoint-specific normalization.
"""
import numpy as np


def match_moments(original, band, variant):
    if variant not in ['dc','rms']:raise ValueError(variant)
    if original.shape!=band.shape or original.ndim!=4:raise ValueError('Require aligned NCHW arrays')
    if isinstance(original,np.ndarray):
        mean=lambda a:a.mean(axis=(-2,-1),keepdims=True)
        sqrt=np.sqrt
        any_=np.any
        where=np.where
    else:
        import torch
        mean=lambda a:a.mean(dim=(-2,-1),keepdim=True)
        sqrt=torch.sqrt
        any_=lambda a:bool(torch.any(a).item())
        where=torch.where
    mu=mean(original);ac=band-mean(band)
    if variant=='dc':return mu+ac
    target=sqrt(mean((original-mu)**2));observed=sqrt(mean(ac**2))
    if any_((observed<=1e-12)&(target>1e-12)):
        raise ValueError('Nonconstant original with zero AC in filtered band; RMS matching undefined')
    divisor=where(observed>1e-12,observed,observed*0+1)
    gain=where(target>1e-12,target/divisor,target*0)
    return mu+ac*gain


def numpy_band(images,radius,family,mode):
    h,w=images.shape[-2:];y=np.fft.fftfreq(h)*h;x=np.fft.fftfreq(w)*w
    d2=y[:,None]**2+x[None,:]**2
    low=(d2<=radius*radius) if family=='hard' else np.exp(-np.log(2)*d2/(radius*radius))
    mask=low if mode=='lp' else 1-low
    return np.fft.ifft2(np.fft.fft2(images,axes=(-2,-1))*mask,axes=(-2,-1)).real


def self_check():
    rng=np.random.default_rng(20261027);x=rng.uniform(.1,.9,size=(6,3,64,64));checks=[]
    for family in ['hard','gaussian']:
        for radius in [5,10]:
            lp=numpy_band(x,radius,family,'lp');hp=numpy_band(x,radius,family,'hp')
            assert np.allclose(lp+hp,x,atol=1e-12)
            restored=match_moments(x,hp,'dc')
            assert np.allclose(restored.mean((-2,-1)),x.mean((-2,-1)),atol=1e-12)
            # Adding DC changes no nonzero-frequency Fourier coefficient.
            old=np.fft.fft2(hp,axes=(-2,-1));new=np.fft.fft2(restored,axes=(-2,-1))
            old[...,0,0]=0;new[...,0,0]=0;assert np.allclose(old,new,atol=1e-10)
            for band in [lp,hp]:
                a=match_moments(x,band,'rms')
                assert np.allclose(a.mean((-2,-1)),x.mean((-2,-1)),atol=1e-12)
                assert np.allclose(a.std((-2,-1)),x.std((-2,-1)),atol=1e-12)
            checks.append(f'{family} radius {radius}: complementarity, DC-only restoration, mean/RMS equality')
    const=np.full((2,3,8,8),.4);assert np.allclose(match_moments(const,const*0,'rms'),const)
    try:match_moments(x,x*0,'rms')
    except ValueError:pass
    else:raise AssertionError('Undefined RMS restoration accepted')
    return {'status':'passed','checks':checks+['constant-channel convention','undefined nonconstant zero-band rejected'],
        'backend':'NumPy float64','gpu_execution':False}


if __name__=='__main__':
    import json
    print(json.dumps(self_check(),indent=2))
