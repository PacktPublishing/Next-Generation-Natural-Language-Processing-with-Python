import collections
import matplotlib.pyplot as plt
import gensim

def plotTopicProjections(model,dictionary,scale=False,plotNegative=False,nTerms=5):
    '''
    Convenience function to plot term importances in topics
    @plotNegative is for models that return -ve term importances
    @scale is either fixed at [-1,1] or autoscaled based on largest importance
    @model is LDA/LSI gensim model object
    '''
    
    topicProjections=model.get_topics()
    
    for n in range(topicProjections.shape[0]):
        #print(i)
        topicTerm=collections.Counter({dictionary[j]:p for j,p in\
                                       enumerate(topicProjections[n,:])})

        most = topicTerm.most_common(nTerms)[::-1]
        least = topicTerm.most_common()[-1*nTerms:]
        
        if not scale:
            plt.xlim(-1,1)
            maxExtent=1
        else:
            maxMost=max([m[1] for m in most])*1.1
            minLeast=min([l[1] for l in least])*1.1
            
            maxMost=topicProjections.max()*1.1
            minMost=topicProjections.min()*1.1
            
            maxExtent=max([abs(minLeast),abs(maxMost)])
            plt.xlim(-1*maxExtent,maxExtent)
                    
        plt.barh(range(nTerms),[m[1] for m in most])
        for i,m in enumerate(most):
            plt.annotate('{:s} ({:.3f})'.format(m[0],m[1]),\
                         xy=(0.1*maxExtent,i-0.1),xycoords='data',fontsize=20)
        
        if not plotNegative:
            if not scale:
                plt.xlim(0,1)
            else:
                plt.xlim(0,maxExtent)
        
        plt.barh(range(nTerms),[l[1] for l in least])
        for i,l in enumerate(least):
            plt.annotate('{:s} ({:.3f})'.format(l[0],l[1]),\
                         xy=(-0.1*maxExtent,i-0.1),xycoords='data',ha='right',fontsize=20)
        plt.axvline(color='grey')
        plt.title('Topic {:d}'.format(n))
        plt.yticks([],[])
        plt.xlabel('Projection')
        plt.show()