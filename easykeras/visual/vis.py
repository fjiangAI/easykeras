__author__ = 'jf'
import matplotlib.pyplot as plt
def plot_curve(history,filename,figtype="acc"):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc)+1)
    if figtype=="acc":
        plt.plot(epochs,acc,'bo',label='Training acc')
        plt.plot(epochs,val_acc,'b',label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(filename, format='png')
    else:
        plt.plot(epochs,loss,'bo',label='Training loss')
        plt.plot(epochs,val_loss,'b',label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(filename, format='png')
'''
使用例子
history=model.fit([inputs_train, queries_train], answers_train,
          batch_size=320,
          epochs=100,
          validation_data=([inputs_test, queries_test], answers_test))
filename="example.png"
plot_curve(history,filename)
'''