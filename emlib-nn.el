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

;; A simple implementation of neural networks.

;;; Code:

(require 'emlib-math)
(require 'dash)

(defun emlib-rand-mat (i j)
  "Generate a matrix of I x J order with random elements."
  (emlib-mat-create (lambda (_ _) (emlib-rand -1 1)) i j))


(defun emlib-layer-create (i h)
  "Create a neural network layer with I inputs and H nodes.

Each layer is represented by the weight matrix, the output
produced during the last forward pass, the last change in the
weight matrix.  We need the change in the weight matrix to make
sure that we can support momentum while updating the weights."
  (let* ((w (emlib-rand-mat h i))
         (dw (emlib-mat-create (lambda (_ _) 0) h i))
         (o (emlib-vec-create (lambda (_) 0) h))
         (eterms (emlib-vec-create (lambda (_) 0) h)))
    (list :weights w
          :delta-weights dw
          :outputs o
          :error-terms eterms)))


(defun emlib-layer-feed (layer inputs)
  "Feed into LAYER a vector of INPUTS.

Updates outputs of the layer. This function doesn't add any new
inputs. It is assumed that a bias term was added if it was
needed.  See `emlib-nn-feed'."
  (let* ((layer-weights (plist-get layer :weights))
         (new-outputs (emlib-mat-mult layer-weights
                                      inputs)))
    ;; Update the outputs vector in the layer.
    (plist-put layer :outputs new-outputs)))


(defun emlib-nn-create (i &rest hlist)
  "Create a neural network with I inputs, HLIST hidden node spec.

HLIST is a list of integers specifying the number of nodes in the
hidden layers. The last number in HLIST specifies the number of
nodes in the output layer."
  (let* ((input-counts (mapcar '1+ (cons i hlist)))
         (dim-pairs (-zip-with 'cons  input-counts hlist))
         (layers (-map (lambda (dim-pair)
                         (emlib-layer-create (car dim-pair)
                                             (cdr dim-pair)))
                       dim-pairs)))
    ;; Making it a property to be able to add meta-data when needed.
    (list :layers layers)))


(defun emlib-nn-feed (network inputs-without-bias)
  "Feed INPUTS to NETWORK updating all the layer outputs.

INPUTS is a sequence of inputs. It is internally converted into a
vector with `emlib-vec-create' after adding a bias term to
it.. This implements the forward pass for a feedforward neural
network.

Note: This function takes care of adding the bias input by
appending a 1 at the end of the inputs vector"
  (let* ((layers (plist-get network :layers))
         (inputs (emlib-vec-from-seq (vconcat inputs-without-bias [1]))))
    (dolist (layer layers)
      (emlib-layer-feed layer inputs)
      (setq inputs (emlib-vec-append-seq (plist-get layer :outputs) [1])))
    (emlib-vec-to-seq (plist-get (car (last layers)) :outputs))))


(provide 'emlib-nn)
;;; emlib-nn.el ends here
