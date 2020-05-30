# Ottimizzazione modello di apprendimento
Ho a disposizione due modelli OLD e VGG
Dopo aver testato i due modelli con dati normazlizzati e con dati standardizzati ho ottenuto i segunti risultati:
|dati|modello|test loss|test accuracy|
|---|---|---|---|
|norm|old drop|1.41|0.60|xx
|st|old drop|1.7|0.59|
|norm|old|3.9834514423544674|0.5528002381324768|
|st|old|4.092885607296658|0.5475062727928162|
|norm|VGG adadelta 0.1 0.95 1e-8|1.2|0.54|
|st|VGG adadelta 0.1 0.95 1e-8|1.9|0.58|
|norm|VGG adadelta 0.1 0.95 1e-8|1.1885047499334358|0.5391473770141602|xx
|st|VGG adadelta 0.1 0.95 1e-8|1.7435963537142316|0.5845639705657959|
|norm|VGG adadelta 1 0.95 1e-6|2.966206067915114|0.5962663888931274|x
|st|VGG adadelta 1 0.95 1e-6|4.09284368070197|0.5898578763008118|
|norm|VGG adam 0.001 0.9 0.999|2.925088699762786|0.5906937718391418|x
|st|VGG adam 0.001 0.9 0.999|3.6050169121823825|0.600724458694458|
|norm|old drop adam 0.001 0.9 0.999|1.584704466734998|0.5901365280151367|xx
|st|old drop adam 0.001 0.9 0.999|1.8006505315168948|0.5965449810028076|
|norm|old adam 0.001 0.9 0.999|3.5713952487808442|0.5622736215591431|
|st|old adam 0.001 0.9 0.999|4.13727425671979|0.5617163777351379

Sceglo di usare dati normalizzati
modelli prescelti:
|dati|modello|test loss|test accuracy|
|---|---|---|---|
|norm|old drop adam 0.001 0.9 0.999|1.584704466734998|0.5901365280151367|xx
|st|old drop adam 0.001 0.9 0.999|1.8006505315168948|0.5965449810028076|
|norm|VGG adadelta 0.1 0.95 1e-8|1.1885047499334358|0.5391473770141602|xx
|st|VGG adadelta 0.1 0.95 1e-8|1.7435963537142316|0.5845639705657959|

controllo funzione non_so_se_ha_effetto():
|dati|modello|test loss|test accuracy|
|---|---|---|---|
|norm|old drop adam 0.001 0.9 0.999|||xx
|norm|VGG adadelta 0.1 0.95 1e-8|1.4607240177854983|0.42490944266319275|
MA sembrava andare però lento vedremo poi

poi riprovare per VGG con standardizzati
ricerca per l1 e l2 con dati norm su old drop adam e vgg adadelta 0.1
eseguo ricerca per learning rate e epoche 32 64 128
```
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
```
## vgg adadelta 0.1
### L1 
tra 0.01, 0.001, 0.0001 -> 0.001
Test loss: 2.0195344994745072
Test accuracy: 0.17442184686660767

### L2
tra 1 0.1 0.01 0.001 0.0001-> 0.01
Test loss: 1.789580599117093
Test accuracy: 0.17442184686660767
### gridsearch for epsilon and learning rate