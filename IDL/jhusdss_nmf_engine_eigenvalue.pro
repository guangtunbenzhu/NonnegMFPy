;+
; NAME:
;   jhusdss_nmf_engine_eigenvalue
; PURPOSE:
;   Given data and NMF basis, calculate coefficient/eigen_values
; CALLING SEQUENCE:
;   jhusdss_nmf_engine_eigenvalue, data, eigen_vectors, weight=weight, eigen_values=eigen_values
; INPUTS:
;   data - [N_spec, N_pix]
;   N_spec - Number of observations/experiments, e.g., number of spectra/galaxies.
;   N_pix  - Number of coordinates/dimensions, e.g., number of wavelengths in a given spectrum
; OPTIONAL INPUTS:
;   weight - [N_spec, N_pix], weight for each data point, e.g., ivar, by default all 1.
;   tolerance - tolerance to stop the iteration, by default 1.d-6
;   maxiters - maximum iterations, by default 1000.
; OPTIONAL INPUTS/OUTPUTS:
;   eigen_values - [N_spec, N_dimension], the eigen_values
;                  Inputs will be treated as initial guess.
; KEYWORDS:
;   silent - shut up
; COMMENTS:
;   -- a truncated version of jhusdss_nmf_engine.pro
; REVISION HISTORY:
;   15-Nov-2011 Guangtun Ben Zhu, JHU
;-

;; a truncated version of jhusdss_nmf_engine.pro
;; Given spectra and nmf basis, use the update rules to get coefficients/eigen_values
pro jhusdss_nmf_engine_eigenvalue, V, W, Weight=Weight, tolerance=tolerance, $
    eigen_values=eigen_values, maxiters=maxiters, silent=silent

;; maximum iterations
;; This default value is 1000 instead of 1200 as in nmf_engine.
;; Don't think it matters
if (n_elements(maxiters) eq 0) then maxiters=1000L
;; default tolerance -- no tolerance, always perform maximum iterations
if (n_elements(tolerance) eq 0) then tolerance=1.d-5

;; V(data[N_spec, N_pix])
if (size(V, /n_dimension) ne 2) then $
   message, "The input data matrix should be in form of [Nk(n_spec), Nn(n_pix)]"
n_spec = (size(V))[1]
n_pix = (size(V))[2]

;if (n_spec lt 10 or n_pix lt 10) then $
;   message, "Either number of spectra < 10 or number of pixels < 10." 
if (n_pix lt 10) then $
   message, "The number of pixels < 10." 

;; Check eigen_vectors
n_dimension = (size(W))[1]
if ((size(W))[2] ne n_pix) then $
   message, "The number of pixels in NMF eigen_vectors does not match your data."

;; Note I am not checking the dimension of weight/ivar
;; Weight[N_spec, N_pix]
if (n_elements(Weight) eq 0) then $
   Weight = fltarr(n_spec, n_pix)+1.

;; check nonnegativity
ilezero = where((V le 0.) and (Weight gt 0.), nlezero)
if (nlezero gt 0) then $
    message, 'jhusdss_nmf_engine requires data to be all-positive'

if (~keyword_set(silent)) then begin
   splog, 'number of spectra: '+strtrim(string(n_spec),2)
   splog, 'number of wavelengths: '+strtrim(string(n_pix),2)
   splog, 'number of eigen_vectors given: '+strtrim(string(n_dimension),2)
   splog, 'maximum iterations will be: '+strtrim(string(maxiters),2)
endif


;; Note Weight(ivar)=1./sigma^2)
VVweight = V*Weight

;; initialization, uniform random number for now
seed = 0.23
if (n_elements(eigen_values) gt 0) then begin
   if ((size(eigen_values))[1] ne n_spec or (size(eigen_values))[2] ne n_dimension) then $
      message, "Your initial guess of eigen_values are in the wrong form."
   H = eigen_values 
endif else H = randomu(seed, n_spec, n_dimension, /double)+1.d-4

err=total((V-W##H)^2*Weight, /double)
eold=1.d+100
iters = 1L

while(iters le maxiters and abs(err-eold)/eold gt tolerance) do begin

   WT = TRANSPOSE(W)
   H_up =  WT##VVweight
   H_down = WT##((W##H)*Weight)

   ;; Note we change H here, meaning when updating W later, we use this updated H
   H = temporary(H)*H_up/H_down

;; We don't update W here
;  HT = TRANSPOSE(H)
;  W_up = VVweight##HT
;  W_down = ((W##H)*Weight)##HT

;  W = temporary(W)*W_up/W_down

   eold = err
   err=total((V-W##H)^2*Weight, /double)

   if (finite(err) eq 0) then $
      message, "NMF failed, likely due to missing data!"

   if (~keyword_set(silent)) then $
      if (iters mod 40 eq 0) then print, err, ' ', eold, ' ', abs(err-eold)/eold, ' ', $
          strtrim(string(iters),2)+'/'+strtrim(string(maxiters),2)

   iters++
endwhile

;; output
if (arg_present(eigen_values)) then eigen_values = H

return

end
