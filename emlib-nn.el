;;; emlib-nn.el --- Neural networks for `emlib' -*- lexical-binding: t; -*-

;; Copyright (C) 2016  Narendra Joshi

;; Author: Narendra Joshi <narendraj9@gmail.com>
;; Keywords: data

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <http://www.gnu.org/licenses/>.

;;; Commentary:

;; A simple implementation of feedforward neural networks.

;;; Code:

(require 'dash)
(require 'emlib-math)


(defun emlib-layer-create (i h)
  "Create a neural network layer with I inputs and H nodes.

Each layer is represented by the weight matrix, the output
produced during the last forward pass, the last change in the
weight matrix.  We need the change in the weight matrix to make
sure that we can support momentum while updating the weights.  An
vector of error terms is also kept in the layer property list to
aid the back propagation step."
  (let* ((w (emlib-rand-mat h i))
         (dw (emlib-mat-create (lambda (_ _) 0) h i))
         (o (emlib-vec-create (lambda (_) 0) h))
         (eterms (emlib-vec-create (lambda (_) 0) h)))
    (list :weights w
          :delta-weights dw
          :outputs o
          :error-terms eterms
          :squashing-fn 'emlib-sigmoid)))


(defun emlib-layer-feed (layer inputs)
  "Feed into LAYER a vector of INPUTS.

Updates outputs of the layer.  This function doesn't add any new
inputs.  It is assumed that a bias term was added if it was
needed.  See `emlib-nn-feed'."
  (let* ((layer-weights (plist-get layer :weights))
         (squashing-fn (plist-get layer :squashing-fn))
         (new-outputs (emlib-mat-map squashing-fn
                                     (emlib-mat-mult layer-weights
                                                     inputs))))
    ;; Update the outputs vector in the layer.
    (plist-put layer :outputs new-outputs)))


(defun emlib-nn-create (i &rest hlist)
  "Create a neural network with I inputs, HLIST hidden node spec.

HLIST is a list of integers specifying the number of nodes in the
hidden layers.  The last number in HLIST specifies the number of
nodes in the output layer."
  (let* ((input-counts (mapcar '1+ (cons i hlist)))
         (dim-pairs (-zip-with 'cons  input-counts hlist))
         (layers (-map (lambda (dim-pair)
                         (emlib-layer-create (car dim-pair)
                                             (cdr dim-pair)))
                       dim-pairs)))
    ;; Making it a property to be able to add meta-data when needed.
    (list :layers layers
          :input-order i
          :hidden-order hlist)))


(defun emlib-nn-feed (network inputs-without-bias)
  "Feed INPUTS to NETWORK updating all the layer outputs.

INPUTS is a sequence of inputs.  It is internally converted into a
vector with `emlib-vec-create' after adding a bias term to
it.. This implements the forward pass for a feedforward neural
network.

Note: This function takes care of adding the bias input by
appending a 1 at the end of the inputs vector

Argument INPUTS-WITHOUT-BIAS is the input sequence without the
bias term."
  (let* ((layers (plist-get network :layers))
         (inputs (emlib-vec-from-seq (vconcat inputs-without-bias [1]))))
    (dolist (layer layers)
      (emlib-layer-feed layer inputs)
      (setq inputs (emlib-vec-append-seq (plist-get layer :outputs) [1])))
    (emlib-vec-to-seq (plist-get (car (last layers)) :outputs))))


(defun emlib--nn-eterms-for-output-layer (network targets)
  "Compute the error terms for output layer of NETWORK.

Note: Current this function is implemently only for sigmoidal
thresholds.  I intend to generalize this to include more squashing
functions.
Argument TARGETS is the target vector."
  (let* ((layers (plist-get network :layers))
         (output-layer (car (last layers)))
         (outputs (plist-get output-layer :outputs)))
    (plist-put output-layer
               :error-terms
               (emlib-vec-create (lambda (i)
                                   (let ((o_i (emlib-vec-get outputs i))
                                         (t_i (emlib-vec-get targets i)))
                                     (* (- 1 o_i) o_i (- t_i o_i))))
                                 (emlib-vec-size outputs)))))


(defun emlib--nn-eterms-backprop (network)
  "Backpropagate the error terms to hidden layer of NETWORK.

Note: We assume that the error terms for the output layer have
been computed."
  (let* ((layers (reverse (plist-get network :layers)))
         (layer-count (length layers)))
    ;; Layer at index 0 is the output layer here.
    (dolist (layer-idx (number-sequence 1 (1- layer-count)))
      ;; The downstream layer comes before the current layer in layers
      (let* ((downstream-layer (nth (1- layer-idx) layers))
             (downstream-weights (plist-get downstream-layer :weights))
             (downstream-eterms (plist-get downstream-layer :error-terms))
             (current-layer (nth layer-idx layers))
             (current-layer-outputs (plist-get current-layer :outputs))
             (current-layer-eterm-count (emlib-vec-size (plist-get current-layer
                                                                   :error-terms)))
             ;; eterms* contains an extra element because of the bias
             ;; unit weights.
             (eterms* (emlib-mat-mult (emlib-mat-transpose downstream-weights)
                                      downstream-eterms)))
        (plist-put current-layer
                   :error-terms
                   (emlib-vec-create
                    (lambda (i)
                      (let ((o_i (emlib-vec-get current-layer-outputs
                                                i)))
                        (* o_i
                           (1- o_i)
                           (emlib-vec-get eterms* i))))
                    current-layer-eterm-count))))))


(defun emlib--nn-weights-update (network input-vector)
  "Update the weights of all units in NETWORK for INPUT-VECTOR.

Note: NETWORK should have corret values for the error terms.
Assuming that the error terms are computed for all the neuron
units, we derive the weight updates and tune NETWORK."
  (let ((layers (plist-get network :layers))
        ;; Add the bias term to the input vector
        (inputs (emlib-vec-append-seq input-vector [1])))
    ;; Iterate through the layers updating weights and delta-weights.
    (dolist (layer layers)
      ;; **TOFIX** Hardcoding a few parameters.
      (let* ((eterms (plist-get layer :error-terms))
             (weights (plist-get layer :weights))
             (weights-dims (emlib-mat-dims weights))
             (delta-weights (plist-get layer :delta-weights))
             ;; Create a new delta-weights matrix
             (new-delta-weights (emlib-mat-scale
                                 (emlib-mat-mult
                                  eterms
                                  (emlib-mat-transpose inputs))
                                 ;; hard-coded learning rate **TOFIX**
                                 0.1)))
        (plist-put layer
                   :weights
                   (emlib-mat-create (lambda (i j)
                                       (+ (emlib-mat-get weights i j)
                                          (emlib-mat-get new-delta-weights i j)
                                          ;; **TOFIX** hard-coding momentum
                                          (* 0.01
                                             (emlib-mat-get delta-weights i j))))
                                     (car weights-dims)
                                     (cdr weights-dims)))
        (plist-put layer
                   :delta-weights
                   new-delta-weights)
        (setq inputs (emlib-vec-append-seq (plist-get layer :outputs) [1]))))))


(defun emlib--nn-backprop (network input-vector target-vector)
  "Perform backprop set for NETWORK given expected TARGETS.
TARGET is a emlib vector.
Argument INPUT-VECTOR is the vector of inputs fed to the network.
Argument TARGET-VECTOR is the expected result of feeding INPUT-VECTOR."
  (emlib--nn-eterms-for-output-layer network target-vector)
  (emlib--nn-eterms-backprop network)
  (emlib--nn-weights-update network input-vector))


(defun emlib-nn-train (network input output &optional)
  "Train NETWORK on example (INPUT, OUTPUT).

INPUT and OUTPUT must be sequences.  They are internally kept as
emlib vectors.  This performs a forward feed step followed by
backprop to tune weights based on a single input using stochastic
gradient descent."
  (let ((input-vector (emlib-vec-from-seq input))
        (target-vector (emlib-vec-from-seq output)))
    ;; Let's feed the input first to change all the unit outputs.
    (emlib-nn-feed network input)
    ;; Now perform the backpropagation step
    (emlib--nn-backprop network input-vector target-vector)))


(provide 'emlib-nn)
;;; emlib-nn.el ends here
