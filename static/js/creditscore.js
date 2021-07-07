/*
Created DateTime Tue, Jun 16 2020, 11:15:16 

@author: Neelam Tavhare
*/

$(document).ready(function(){

    let fileToUpload;
    
    // File upload drag/drop section
    let currentFile = {};
    let isAdvancedUpload = function () {
        let div = document.createElement('div');
        return (('draggable' in div) || ('ondragstart' in div && 'ondrop' in div)) && 'FormData' in window && 'FileReader' in window;
    }();
    let afterFileUpload = function (preview, name) {
        $(`#file-section .file-display-container img`).attr('src', preview);
        $('.file-name').text(name);
        $(`#file-section .drop-zone-container`).removeClass('d-flex').addClass('d-none');
        $(`#file-section .file-display-container`).removeClass('d-none').addClass('d-flex');
    }
    let dropZone = $('.drop-zone');
    if (isAdvancedUpload) {
        dropZone.addClass('has-advanced-upload');
        $(document).on('drag dragstart dragend dragover dragenter dragleave drop', '.drop-zone', function (e) {
            e.preventDefault();
            e.stopPropagation();
        }).on('dragover dragenter', '.drop-zone', function () {
            $(this).addClass('is-dragover');
        }).on('dragleave dragend drop', '.drop-zone', function () {
            $(this).removeClass('is-dragover');
        }).on('drop', '.drop-zone', function (e) {
			fileToUpload=e.originalEvent.dataTransfer;
			var reader = new FileReader();
			reader.onload = function (e) {
				afterFileUpload(e.target.result, name);
			}
			reader.readAsDataURL(fileToUpload.files[0]);
        });
    }
    $('#drop-zone-file').on('change', function (event) {
		fileToUpload=event.target;
		var reader = new FileReader();
		reader.onload = function (e) {
			afterFileUpload(e.target.result, name);
		}
        reader.readAsDataURL(fileToUpload.files[0]);
    });
    $('.upload-section .icon-refresh').click(function () {
        $(`.file-display-container img`).attr('src', '');
        $(`#file-section .drop-zone-container`).removeClass('d-none').addClass('d-flex');
        $(`.file-display-container`).removeClass('d-flex').addClass('d-none');
        $('#resultDiv').hide();
		$('.output-status').text('Please upload a file').show();
        $('.loader').removeClass('d-flex').addClass('d-none');
        $('.status-container').removeClass('d-none').addClass('d-flex');
    });
	
    $('#submitButton').on('click', function (event) {
		$('#drop-zone-label').hide();
		$('#icon-upload').hide();
		$('#uploader-div').hide();
		

		processData();
    });

    function processData(){  
        
        if (fileToUpload.files && fileToUpload.files[0]) {
            let name = fileToUpload.files[0].name;
			let fileExtension=name.split(".")[name.split(".").length-1];
			
			if(fileExtension.toLowerCase()!="csv"){
				$('.output-status').text('Please upload csv file only. Invalid file extension!').show();
				return;
			}
			
            let base = ".";
            let url = base + '/creditscore';
            let formData = new FormData();
            formData.append('inputDataFile', fileToUpload.files[0]);
            console.log(formData);
            $.ajax({
            
                url: url,
                type: 'post',
                data: formData,
                processData: false,
                contentType: false,
                beforeSend: function () {
                    $('.output-status').hide();
					$('#resultDiv').hide();
					$('#drop-zone-label').hide();
                    $('.loader').removeClass('d-none').addClass('d-flex');
                },
                success: function (response, status, xhr) {/*
					
						*/
						/*$('.output-status').text('Processing the data uploaded... download the file once generated').show();*/
						console.log(xhr.getResponseHeader("content-disposition"));
						console.log(status);
                        console.log(response);
                        // The actual download
                        $('.output-status').hide();
                        var blob = new Blob([response], { type: 'text/csv' });
                        var link = document.createElement('a');
                        link.href = window.URL.createObjectURL(blob);
                        link.download = "export.csv";
                
                        document.body.appendChild(link);
                
                        link.click();
                
                        document.body.removeChild(link);
						
                        
                    
                },
                error: function (xhr, status, error) {
					var JSONresp = JSON.parse(xhr.responseText);
                    $('.status-container').removeClass('d-none').addClass('d-flex');
					$('.output-status').text('Error: '+JSONresp.message).show();
                },
                complete: function (xhr, status) {

                    $('.loader').removeClass('d-flex').addClass('d-none');
                }
            });
        
        }
        
    }
});