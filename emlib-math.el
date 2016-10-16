;;; emlib-math.el --- Mathematical functions for emlib -*- lexical-binding: t; -*-

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

;; This file contains functions for creating, manipulating and
;; operating on matrices.  Moreover this would be the home for all the
;; mathematical functions needed to be implemented for the rest of the
;; package.

;;; Code:

(eval-when-compile
  (require 'cl-lib))

(require 'calc-ext)

(defun emlib-mat-dims (matrix)
  "Return dimens of MATRIX as in a cons cell."
  (car matrix))

(defun emlib-mat-create (element-function m n)
  "Call ELEMENT-FUNCTION with indices to generate matrix of order M x N.

ELEMENT-FUNCTION takes two numbers, i.e. the indices for the
position of the element inside the matrix and returns a value to
be placed there.

Note: currently, matrices are all two dimensional."
  (let* ((aux-elem-fn (lambda (position)
                        (funcall element-function
                                 (/ position n)
                                 (mod position n))))
         (mat-size (* m n))
         (mat-as-vec (make-vector mat-size 0)))
    (dotimes (i mat-size)
      (aset mat-as-vec i (funcall aux-elem-fn i)))
    (cons (cons m n) mat-as-vec)))


(defun elmlib-vec-create (element-fn size)
  "Create column vector (with ELEMENT-FN) of SIZE."
  (emlib-mat-create element-fn (cons size 1)))


(defun emlib-mat-get (matrix i j)
  "Query MATRIX for element at INDICES.
Argument I row number.
Argument J column number."
  (let* ((mat-as-vec (cdr matrix))
         (dims (emlib-mat-dims matrix))
         (cols (cdr dims)))
    (aref mat-as-vec (+ (* i cols) j))))

(defun emlib-vec-get (v i)
  "Return vector V's Ith element."
  (emlib-mat-get v i 0))

(defun emlib-mat-op (op a b)
  "Apply operation OP to respectivve elements of A and B."
  (let* ((a-dims (emlib-mat-dims a))
         (b-dims (emlib-mat-dims b))
         (compose-fn (lambda (x y)
                       (funcall op
                                (emlib-mat-get a x y)
                                (emlib-mat-get b x y)))))
    (if (equal a-dims b-dims)
        (emlib-mat-create compose-fn (car a-dims) (cdr a-dims))
      (error "Order of the two matrices must be equal"))))

(defun emlib-mat-to-string (mat &optional elem-width)
  "Return string representation for matrix MAT.
Optional argument ELEM-WITH when non-nil specifies the width of
printed version of each matrix element.
Optional argument ELEM-WIDTH space occupied by elemnt in string."
  (let* ((dims (emlib-mat-dims mat))
         (rows (car dims))
         (cols (cdr dims)))
    (with-output-to-string
      (dotimes (i rows)
        (dotimes (j cols)
          (princ (format (concat "%"
                                 (number-to-string (or elem-width 5))
                                 "s")
                         (emlib-mat-get mat i j))))
        (princ "\n")))))

(defun emlib-mat-add (a b)
  "Add matrices A and B."
  (emlib-mat-op '+ a b))

(defun emlib-mat-sub (a b)
  "Compute A - B."
  (emlib-mat-op '- a b))

(defun emlib-mat-scale (mat factor)
  "Scale very element of MAT by FACTOR."
  (let* ((dims (emlib-mat-dims mat))
         (rows (car dims))
         (cols (cdr dims)))
    (emlib-mat-create (lambda (i j)
                        (* factor (emlib-mat-get mat i j)))
                      rows
                      cols)))

(defun emlib-mat-identity (size)
  "Return an identity matrix of order equal to SIZE."
  (emlib-mat-create (lambda (i j)
                      (if (= i j)
                          1
                        0))
                    size
                    size))


(defun emlib-mat-mult (a b)
  "Multiply matrix A by matrix B."
  (let* ((a-dims (emlib-mat-dims a))
         (b-dims (emlib-mat-dims b))
         (a-rows (car a-dims))
         (a-cols (cdr a-dims))
         (b-rows (car b-dims))
         (b-cols (cdr b-dims)))
    (if (not (= a-cols b-rows))
        (error "Invalid orders for matrices: %s %s " a-dims b-dims)
      (emlib-mat-create (lambda (i j)
                          (let ((k-range (number-sequence 0 (1- a-cols))))
                            (apply '+
                                   (mapcar (lambda (k) (* (emlib-mat-get a i k)
                                                          (emlib-mat-get b k j)))
                                           k-range))))
                        a-rows
                        b-cols))))

(provide 'emlib-math)
;;; emlib-math.el ends here
