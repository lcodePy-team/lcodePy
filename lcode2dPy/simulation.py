class DefaultDiag:

    class Ez:
        def __init__(self):
            pass

        def dump(self, *args):
            pass 

        def d_xi(self, *args):
            pass

    class Another:
        def __init__(self):
            pass

        def dump(self, *args):
            pass


class Diagnostics:
    
    def __init__(self, config, \
                list_t = [DefaultDiag.Another()], \
                list_xi = [DefaultDiag.Ez()]):
        self.list_t  = list_t
        self.list_xi = list_xi
    
    def t_step(self, *args):
        for diag in self.list_t:
            diag.dump(*args)
        for diag in self.list_xi:
            diag.dump(*args)
    
    def xi_step(self, *args):
        for diag in self.list_xi:
            diag.d_xi(*args)

class Simulation():
    def __init__(self, config, diag):
        self.config = config
        self.cur_t = 0
        self.PAS = PusherAndSolver(config)
        self.diag = diag
        # TODO BEAM 

    def steps(self, N):
        t_start = self.cur_t
        t_end = self.cur_t+N*self.time_step
        
        for t in range(t_start, t_end):
            
            particles, fields = self.PAS.step_dt(particles, fields, beam_source, beam_drain, t, diag)

            # Я бы убрал dump внуть Pusher&Solver
            if time_for_diag:
                for diag in self.list_t:
                    diag.dump()
                for diag in self.list_xi:
                    diag.dump()


