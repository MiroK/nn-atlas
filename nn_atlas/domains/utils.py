import numpy as np


def line_line(ref, target, f=lambda x: x):
    '''Mapping from reference line to target line'''
    tol = 1E-13
    # [A, B] => P = A + t*(B-A)  for t in (0, 1)
    # [a, b] = a + f(t)*(b-a) for t in (0, 1)
    assert np.abs(f(0)) < tol and np.abs(f(1)-1) < tol
    # It should also be invergible on (0, 1) but I don't check
    # for that
    def mapping(P, f=f, ref=ref, target=target):
        A, B = ref
        t = np.dot(P-A, B-A)/np.dot(B-A, B-A)
        assert -tol < t < 1+tol
        
        a, b = target
        return a + (b-a)*f(t)

    return mapping


def line_circleArc(ref, target, f=lambda x: x, orientation=0):
    '''Mapping from reference line to target circle arc'''
    tol = 1E-13

    a, center, b = target
    assert abs(np.linalg.norm(a-center) - np.linalg.norm(b-center)) < tol, \
        (np.linalg.norm(a-center), np.linalg.norm(b-center))
    # Let's build parametrization of cirlce arc
    radius = np.linalg.norm(a-center)
    alpha = -2*np.arcsin(0.5*np.linalg.norm(b-a)/radius)

    # For counterclockwise
    if orientation == 0:
        R = lambda t, alpha=alpha: np.array([[np.cos(t*alpha), np.sin(t*alpha)],
                                             [-np.sin(t*alpha), np.cos(t*alpha)]])
    else:
        R = lambda t, alpha=alpha: np.array([[np.cos(t*alpha), -np.sin(t*alpha)],
                                             [np.sin(t*alpha), np.cos(t*alpha)]])    

    def mapping(P, f=f, ref=ref, center=center, a=a, R=R):
        A, B = ref
        t = np.dot(P-A, B-A)/np.dot(B-A, B-A)
        assert -tol < t < 1+tol

        return center + np.dot(R(t), a - center)

    return mapping

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    d = 2
    ref = np.array([[0.5, -1],
                    [0, -1]])
    target = np.array([[0.3, 0.0],
                       [0.25, 0.1],
                       [0.2, 0.0],
                       ])

    target = np.array([[ 0.        ,  0.        ],
                       [-0.41143783,  0.91143783],
                       [ 0.5       ,  0.5       ]])


    # target = np.array([[0.2, 0.0],
    #                    [0.25, 0.1],
    #                    [0.3, 0.0],
    #                    ])

    
    f = line_circleArc(ref, target)

    fig, ax = plt.subplots()
    ax.plot(ref.T[0], ref.T[1], 'r')
    ax.plot(*ref[0], 'ro')    

    ax.plot(*target[0], 'ob')
    ax.plot(*target[1], 'xb')
    ax.plot(*target[2], 'xb')        

    A, B = ref
    t = np.linspace(0, 1, 20)
    for ti in t:
        x_ref = A + ti*(B-A)
        y_ref = f(x_ref)

        seg = np.array([x_ref, y_ref])
        ax.plot(seg.T[0], seg.T[1], 'k')

    ax.axis('equal')
    plt.show()
