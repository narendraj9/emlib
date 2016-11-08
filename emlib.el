;;; emlib.el --- A Machine Learning library for Emacs

;; Copyright (C) 2016  Narendra Joshi

;; Author: Narendra Joshi <narendraj9@gmail.com>
;; URL: https://github.com/narendraj9/emlib.git
;; Version: 0.1
;; Keywords: data, ai, neural networks, ml
;; Package-Requires: ((dash "2.13.0"))

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

;; A library for experimenting with Machine Learning models inside
;; Emacs. If you are looking for performance, this might not be the
;; best place. Or you might take this as a challenge and make the code
;; performant. Cheers! :)
;;

;;; Code:

(require 'emlib-nn)

(provide 'emlib)
;;; emlib.el ends here
