class SampleObjectiveFunction():

    def __init__(self, pipeline, sample_size, w0, df, hft, mdist):
        self.pipeline = pipeline
        self.sample_size = sample_size
        self.w0 = w0
        self.df = df
        self.hft = hft
        self.mdist = mdist

    def evaluate(self, w):
        pupdate = self.pipeline.remap(w, self.w0[1])
        models = []
        sample = self.df.sample(n=self.sample_size)
        for i, row in sample.iterrows():
            g = self.hft.transform(row)
            g = pupdate.transform(g)
            g.id = i
            models.append(g)

        distances = compute_distances(models, self.mdist, [1, 0])
        c = 1 - np.mean(distances)
        return c
