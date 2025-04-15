class MachineData():
    def __init__(self, Rs,
                 Ld,
                 Lq,
                 psi_f,
                 p):
        self.Rs = Rs
        self.Ld = Ld
        self.Lq = Lq
        self.psi_f = psi_f
        self.p = p

    @property
    def chi(self):
        """
        Computes the saliency factor.

        Returns
        -------
        float
            saliency factor chi as in [1]

        References
        -------
        TODO: Complete reference
        [1] AED Book
        """
        return (self.Lq - self.Ld)/(2*self.Ld)
