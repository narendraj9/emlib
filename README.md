emlib
=====

A humble Machine Learning library. Currently, there is a simple implementation
of feedforward networks with learning rate, momemtum and a few parameters
hard-coded. The API would be changed in the future. The idea behind sharing it
is to get more people involved in the development.


## Usage

```elisp
(require 'emlib)

;; Let's define some data with three-dimensional inputs and a scalar
;; output.
(setq data '(([1 0 0] . [0]) ([1 0 1] . [1]) ([1 1 1] . [0])))


;; Now, let's create and train a neural network with 4 nodes in a
;; hidden layer.
(let ((my-nn (emlib-nn-create 3 4 1)))
  (dotimes (i 1000)
    (dolist (input-output data)
      (message "Feeding to the network <%s, %s> "
               (car input-output)
               (cdr input-output))
      (emlib-nn-train my-nn (car input-output) (cdr input-output)))))

(message "Output for [1 0 0]: %s" (emlib-nn-feed my-nn [1 0 0]))


NOTE: The API is definitelly going to change as this project is in its early
stage. The idea behind sharing this is to make more people involved so that it
 is better than what I can make it alone.

```
