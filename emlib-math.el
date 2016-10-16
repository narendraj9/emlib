;;; emlib-math.el --- Mathematical functions for emlib  -*- lexical-binding: t; -*-

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
;; operating on matrices. Moreover this would be the home for all the
;; mathematical functions needed to be implemented for the rest of the
;; package.

;;; Code:


(defun emlib-mat-create [element-function dimensions]
  "Create a matrix of DIMENSIONS by calling ELEMENT-FUNCTION.

ELEMENT-FUNCTION takes a sequence of numbers, i.e. the indices
for the position of the element inside the matrix and returns a
value to be placed there.
DIMENSIONS is a sequence of numbers describding the order of the matrix."
  (ignore))

(provide 'emlib-math)
;;; emlib-math.el ends here
