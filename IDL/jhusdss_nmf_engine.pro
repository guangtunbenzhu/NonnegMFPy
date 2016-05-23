;+
; NAME:
;   jhusdss_nmf_engine
; PURPOSE:
;   non-negative PCA routine 
; CALLING SEQUENCE:
;   jhusdss_nmf_engine, data, weight=weight, n_dimension=n_dimension, tolerance=tolerance, 
;   eigen_vectors=eigen_vectors, eigen_values=eigen_values, maxiters=maxiters, /silent
; INPUTS:
;   data - [N_spec, N_pix]
;   N_spec - Number of observations/experiments, e.g., number of spectra/galaxies.
;   N_pix  - Number of coordinates/dimensions, e.g., number of wavelengths in a given spectrum
; OPTIONAL INPUTS:
;   weight - [N_spec, N_pix], weight for each data point, e.g., ivar, by default all 1.
;   n_dimension - number of basis/eigen vectors sought, by default 10
;   tolerance - tolerance to stop the iteration, by default 1.d-6
;   maxiters - maximum iterations, by default 1000.
;   qapath - path for the qaplot, by default jhusdss_get_path(/nmfqso)+'/QAplot_temp'
; OPTIONAL INPUTS/OUTPUTS:
;   eigen_vectors - [N_dimension, N_pix], the basis/eigen vectors. Inputs will be treated as initial guess.
;   eigen_values - [N_spec, N_dimension]. Inputs will be treated as initial guess.
; KEYWORDS:
;   silent - shut up
;   qaplot - make a qaplot showing how W/H change with iteration
; REFERENCES:
;   -- Lee & Seung (2001) "Algorithms for Non-negative Matrix Factorization"
;   -- Blanton & Roweis, Kcorrect paper, ApJ, 133, 734 (2007)
; COMMENTS:
;   -- V(data[N_spec, N_pix]) = W(eigen_vector[N_dimension, N_pix])##H(eigen_value[N_spec, N_dimension])
;      Find W and H
;   -- Note W and H are switched between the above two references,
;      and we are sticking with Lee & Seung(2001) in which W is the basis/eigen vector matrix
;      and H is the weight/eigen value vector matrix, but the math stays the same
;   -- Note in IDL, the definition of row and column ([column, row]) is the opposite to the convention
;      in mathematical convention ([row, column])
;      In math convention: V[n, m] = W[n, r]H[r, m]
;      In IDL convention: V[m, n] = W[r, n]##H[m, r]
;   -- We use n:N_pix, m:N_spec, r:N_eigen, V:data, W:eigen vectors, H:eigen values
;   -- Note within one iteration, we need to use the first updated matrix (either H or W) to 
;      update the other one (W or H), because the nondecreasing theorems/rules only work for
;      the update of one matrix. See the Lee & Seung 2001 paper.
;   -- See also nmf_sparse.pro by Mike Blanton in IDLutils
; BUGS:
;   -- No bugs so far :-P
; REVISION HISTORY:
;   15-Nov-2011 Guangtun Ben Zhu, JHU
;               -- remove the projection mode
;               -- added weight
;-

pro jhusdss_nmf_engine, V, Weight=Weight, n_dimension=n_dimension, tolerance=tolerance, $
    eigen_vectors=eigen_vectors, eigen_values=eigen_values, maxiters=maxiters, $
    silent=silent, qaplot=qaplot, qapath=qapath, qafile=qafile

;; maximum iterations
if (n_elements(maxiters) eq 0) then maxiters=1000L
;; default tolerance -- no tolerance, always perform maximum iterations
if (n_elements(tolerance) eq 0) then tolerance=1.0d-5

;; V(data[N_spec, N_pix])
if (size(V, /n_dimension) ne 2) then $
   message, 'The input data matrix should be in form of [Nk(n_spec), Nn(n_pix)]'
n_spec = (size(V))[1]
n_pix = (size(V))[2]

if (n_spec lt 10 or n_pix lt 10) then $
   message, "Either number of spectra < 10 or number of pixels < 10." 


;; Note I am not checking the dimension of weight/ivar
;; Weight[N_spec, N_pix]
if (n_elements(Weight) eq 0) then $
   Weight = fltarr(n_spec, n_pix)+1.

;; check nonnegativity
ilezero = where((V le 0.) and (Weight gt 0.), nlezero)
if (nlezero gt 0) then $
    message, 'jhusdss_nmf_engine requires data to be all-positive'

;; how many nmf eigenvectors
if (n_elements(n_dimension) eq 0) then begin
    n_dimension = 12
    if (~keyword_set(silent)) then $
       print, "You didn't give me the number of eigenvector desired, use the default: "+strtrim(string(n_dimension),2)
endif

if (~keyword_set(silent)) then begin
   splog, 'number of spectra: '+strtrim(string(n_spec),2)
   splog, 'number of wavelengths: '+strtrim(string(n_pix),2)
   splog, 'number of eigen_vectors sought: '+strtrim(string(n_dimension),2)
   splog, 'maximum iterations will be: '+strtrim(string(maxiters),2)
endif


;; Note Weight(ivar)=1./sigma^2)
VVweight = V*Weight

;; initialization, uniform random number for now
seed = 0.23
if (n_elements(eigen_vectors) gt 0) then W = eigen_vectors $
else W = randomu(seed, n_dimension, n_pix, /double)+1.d-4
if (n_elements(eigen_values) gt 0) then H = eigen_values $
else H = randomu(seed, n_spec, n_dimension, /double)+1.d-4

err=total((V-W##H)^2*Weight, /double)
eold=1.d+100
iters = 1L

;; randomly save 5*2 pixels/spectra for qaplots
if (keyword_set(qaplot)) then begin
   nqa_pix = 5
   nqa_spec = 5
   if (nqa_spec gt n_spec) then message, "Your training set is less than "+strtrim(string(nqa_spec),2)+"!"
   iqa_pix = floor(randomu(seed, nqa_pix)*(n_pix-1))
   iqa_spec = floor(randomu(seed, nqa_spec)*(n_spec-1))
   iqa_pix = [iqa_pix, iqa_pix+1]
   iqa_spec = [iqa_spec, iqa_spec+1]
   wqa = dblarr(n_dimension, nqa_pix*2, maxiters)
   hqa = dblarr(nqa_spec*2, n_dimension, maxiters)
endif

while(iters le maxiters and abs(err-eold)/eold gt tolerance) do begin

   WT = TRANSPOSE(W)
   H_up =  WT##VVweight
   H_down = WT##((W##H)*Weight)

   ;; Note we change H here, meaning when updating W later, we use this updated H
   H = temporary(H)*H_up/H_down

   HT = TRANSPOSE(H)
   W_up = VVweight##HT
   W_down = ((W##H)*Weight)##HT

   W = temporary(W)*W_up/W_down

   eold = err
   err=total((V-W##H)^2*Weight, /double)

   if (finite(err) eq 0) then $
      message, "NMF failed, likely due to missing data!"

   if (~keyword_set(silent)) then $
      if (iters mod 40 eq 0) then print, err, ' ', eold, ' ', abs(err-eold)/eold, ' ', $
          strtrim(string(iters),2)+'/'+strtrim(string(maxiters),2)

   if (keyword_set(qaplot)) then begin
       wqa[*, *, iters-1] = W[*,iqa_pix]
       hqa[*, *, iters-1] = H[iqa_spec, *]
   endif

   iters++
endwhile

;; output
if (arg_present(eigen_vectors)) then eigen_vectors = W
if (arg_present(eigen_values)) then eigen_values = H

;; QAplot?
if (keyword_set(qaplot)) then begin
   if (n_elements(qapath) eq 0) then qapath = jhusdss_get_path(/nmfqso)+'/QAplot_temp'
   if (n_elements(qafile) eq 0) then qafile = 'nmf_basis_qaplot.ps'
   thick=8
   charthick=3
   charsize=1.5
   k_print, filename=qapath+'/'+qafile, axis_char_scale=1.3, xsize=10, ysize=12
      !p.multi = [0,2,nqa_pix]
      !x.margin = 0
      !y.margin = 0
      xtitle = 'iterations'
      wytitle = 'eigen_vector value'
      hytitle = 'eigen_value value'

      for idim=0, 2 do begin
          for i=0L, nqa_pix-1L do begin
              djs_plot, wqa[idim,i,*], thick=thick, xthick=thick, ythick=thick, $
                  xtickformat='(A1)'
              djs_xyouts, !x.crange[0]+0.8*(!x.crange[1]-!x.crange[0]), $
                  !y.crange[0]+0.8*(!y.crange[1]-!y.crange[0]), $
                  'W'+strtrim(string(iqa_pix[i]),2), charthick=charthick, charsize=charisze
              if (i eq nqa_pix-1) then $
                  djs_axis, xaxis=0, xtitle=xtitle, charthick=charthick, charsize=charsize
              djs_plot, wqa[idim,i+nqa_pix,*], thick=thick, xthick=thick, ythick=thick, $
                  xtickformat='(A1)', ytickformat='(A1)'
              if (i eq nqa_pix-1) then $
                  djs_axis, xaxis=0, xtitle=xtitle, charthick=charthick, charsize=charsize
          endfor
      endfor

      for idim=0, 2 do begin
          for i=0L, nqa_spec-1L do begin
              djs_plot, hqa[i, idim,*], thick=thick, xthick=thick, ythick=thick, $
                  xtickformat='(A1)'
              djs_xyouts, !x.crange[0]+0.8*(!x.crange[1]-!x.crange[0]), $
                  !y.crange[0]+0.8*(!y.crange[1]-!y.crange[0]), $
                  'H'+strtrim(string(iqa_spec[i]),2), charthick=charthick, charsize=charisze
              if (i eq nqa_spec-1) then $
                  djs_axis, xaxis=0, xtitle=xtitle, charthick=charthick, charsize=charsize
              djs_plot, hqa[i+nqa_spec, idim, *], thick=thick, xthick=thick, ythick=thick, $
                  xtickformat='(A1)', ytickformat='(A1)'
              if (i eq nqa_spec-1) then $
                  djs_axis, xaxis=0, xtitle=xtitle, charthick=charthick, charsize=charsize
          endfor
      endfor

   k_end_print
endif

return

end
