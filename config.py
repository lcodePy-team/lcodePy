# Sets parameters of test 1:
config = {
    'geometry': '3d',
    'processing-unit-type': 'cpu',
    'window-width-step-size': 0.05,
    'window-width': 25,

    'window-length': 1000, # Not including the head!!!
    'xi-step': 0.05,

    'time-limit': 1e-10,
    'time-step': 1e-10,

    'plasma-particles-per-cell': 1,

    'filter-window-length': 5,
    'filter-polyorder': 3,
    'filter-coefficient': 0.1,

    'noise-reductor-amplitude': 0,
}