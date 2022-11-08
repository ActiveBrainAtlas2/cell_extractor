import matplotlib.pyplot as plt

class Plotter:
    """class for plotting detection results and sore distributions"""

    def create_plot_image_array(self,images,ncol = 5,titles=None,scatters = None,subplot_size = 2):
        """generate array of plots to show individual detections"""
        nimage = len(images) 
        nrow = nimage//ncol+1
        f = plt.figure(figsize = [subplot_size*ncol,subplot_size*nrow])
        for i in range(nimage):
            ax = plt.subplot(nrow,ncol,i+1)
            ax.imshow(images[i])
            if type(titles) != type(None):
                ax.set_title(titles[i])
            if type(scatters)!=type(None):
                ax.scatter(scatters[i][1],scatters[i][0],color='r')
        return f

    def plot_image_array(self,images,ncol = 5,titles=None,scatters = None,subplot_size = 2):
        """plot an array of images with cell location"""
        self.create_plot_image_array(images,ncol = ncol,titles=titles,scatters = scatters,subplot_size = subplot_size)
        plt.show()
    
    def save_plot_image_array(self,images,save_path,ncol = 5,titles=None,scatters = None,subplot_size = 2):
        """save the figure array to a path"""
        self.create_plot_image_array(images,ncol = ncol,titles=titles,scatters = scatters,subplot_size = subplot_size)
        plt.savefig(save_path)

class DetectionPlotter(Plotter):

    def plot_examplei(self,examplei = 0):
        """plots the image of a specific example

        :param examplei: _description_, defaults to 0
        :type examplei: int, optional
        """
        assert(hasattr(self, 'Examples'))
        examplei = self.Examples[examplei][0]
        ch1 = examplei['image_CH1']
        ch3 = examplei['image_CH3']
        plt.imshow(ch1)
        plt.show()
        plt.imshow(ch3)
        plt.show()
    
    def plot_examples(self,examples,channel = 3):
        """plot image of all examples

        :param examples: _description_
        :type examples: _type_
        :param channel: _description_, defaults to 3
        :type channel: int, optional
        """
        nexamples = len(examples)
        images = [examplei[f'image_CH{channel}'] for examplei in examples]
        titles = ['%d, %d'%(i,examples[i]['area']) for i in range(nexamples)]
        self.plot_image_array(images,titles = titles)
