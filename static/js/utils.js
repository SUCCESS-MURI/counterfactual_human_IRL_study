function downloadCSV(csv, filename) { 
    var csvFile; 
    var downloadLink; 

    //Retrieve csv file from experiment 
    csvFile = new Blob([csv], {type: "text/csv"}); 

    downloadLink = document.createElement( "a"); 

    //Retreive file name 
    downloadLink.download = filename; 

    //create a link to the file 
    downloadLink.href = window.URL.createObjectURL(csvFile); 

    downloadLink.style.display = 'none'; 

    //add link to the DOM 
    document.body.appendChild(downloadLink); 

    downloadLink.click(); 


}