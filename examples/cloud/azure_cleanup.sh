## Azure nic
#az network nic list --query "[?contains(@.name, 'jerry')==\`true\`].name"

# rgname=jerry-assisted-reward-design
rgname=assistive-gym

## Delete used NIC
for nicname in `az network nic list --query "[? contains(name, 'doodad')].name"  -o tsv`;
do
    echo Deleting NIC ${nicname}
    az network nic delete --resource-group ${rgname} --name ${nicname} --no-wait
done


## Delete used VNET
for vnetname in `az network vnet list --query "[? contains(name, 'doodad')].name"  -o tsv`;
do
    echo Deleting VNET ${vnetname}
    az network vnet delete --resource-group ${rgname} --name ${vnetname}
done


## Delete used Disk
for diskname in `az disk list --query "[? contains(name, 'doodad')].name"  -o tsv`;
do
    echo Deleting disk ${diskname}
    az disk delete --resource-group ${rgname} --name ${diskname} --no-wait -y
done



