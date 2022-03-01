/**
 * jspsych-html-button-response
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["likert-sort"] = (function() {

    var plugin = {};
  
    jsPsych.pluginAPI.registerPreload('video-keyboard-response', 'stimulus', 'video');
  
  
    plugin.info = {
      name: 'likert-sort',
      description: '',
      parameters: {
        //SECONDARY CHECK
        sources: {
          type: jsPsych.plugins.parameterType.VIDEO,
          pretty_name: 'Video',
          default: undefined,
          description: 'The video file to play.'
        },
        checkChoices: {
          type: jsPsych.plugins.parameterType.KEYCODE,
          pretty_name: 'Choices',
          array: true,
          default: jsPsych.ALL_KEYS,
          description: 'The keys the subject is allowed to press to respond to the stimulus.'
        },
        width: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Width',
          default: '',
          description: 'The width of the video in pixels.'
        },
        height: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Height',
          default: '',
          description: 'The height of the video display in pixels.'
        },
        autoplay: {
          type: jsPsych.plugins.parameterType.BOOL,
          pretty_name: 'Autoplay',
          default: true,
          description: 'If true, the video will begin playing as soon as it has loaded.'
        },
        controls: {
          type: jsPsych.plugins.parameterType.BOOL,
          pretty_name: 'Controls',
          default: false,
          description: 'If true, the subject will be able to pause the video or move the playback to any point in the video.'
        },

        //PRIMARY QUESTION STIMULUS + QUESTION
        stimuli: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Stimuli',
            default: undefined,
            array: true,
            description: 'Images to be displayed.'
          },
          stim_height: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Stimulus height',
            default: 100,
            description: 'Height of images in pixels.'
          },
          stim_width: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Stimulus width',
            default: 100,
            description: 'Width of images in pixels'
          },
          sort_area_height: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Sort area height',
            default: 800,
            description: 'The height of the container that subjects can move the stimuli in.'
          },
          sort_area_width: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Sort area width',
            default: 800,
            description: 'The width of the container that subjects can move the stimuli in.'
          },
          prompt: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Prompt',
            default: null,
            description: 'It can be used to provide a reminder about the action the subject is supposed to take.'
          },
          prompt_location: {
            type: jsPsych.plugins.parameterType.SELECT,
            pretty_name: 'Prompt location',
            options: ['above','below'],
            default: 'above',
            description: 'Indicates whether to show prompt "above" or "below" the sorting area.'
          },


        //SLIDER RESPONSE 
        min: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Min slider',
            default: 0,
            description: 'Sets the minimum value of the slider.'
        },
        max: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Max slider',
            default: 100,
            description: 'Sets the maximum value of the slider',
        },
        start: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Slider starting value',
            default: 50,
            description: 'Sets the starting value of the slider',
        },
        step: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Step',
            default: 1,
            description: 'Sets the step of the slider'
        },
        labels: {
            type: jsPsych.plugins.parameterType.HTML_STRING,
            pretty_name:'Labels',
            default: [],
            array: true,
            description: 'Labels of the slider.',
        },
        slider_width: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name:'Slider width',
            default: null,
            description: 'Width of the slider in pixels.'
        },
        require_movement: {
            type: jsPsych.plugins.parameterType.BOOL,
            pretty_name: 'Require movement',
            default: false,
            description: 'If true, the participant will have to move the slider before continuing.'
        },


        //LIKERT SCALE
        questions: {
          type: jsPsych.plugins.parameterType.COMPLEX,
          array: true,
          pretty_name: 'Questions',
          nested: {
            
            prompt: {
              type: jsPsych.plugins.parameterType.STRING,
              pretty_name: 'Prompt',
              default: undefined,
              description: 'Questions that are associated with the slider.'
            },
            labels: {
              type: jsPsych.plugins.parameterType.STRING,
              array: true,
              pretty_name: 'Labels',
              default: undefined,
              description: 'Labels to display for individual question.'
            },
            required: {
              type: jsPsych.plugins.parameterType.BOOL,
              pretty_name: 'Required',
              default: false,
              description: 'Makes answering the question required.'
            },
            name: {
              type: jsPsych.plugins.parameterType.STRING,
              pretty_name: 'Question Name',
              default: '',
              description: 'Controls the name of data values associated with this question'
            }
          }
        },
        scale_width: {
          type: jsPsych.plugins.parameterType.INT,
          pretty_name: 'Scale width',
          default: null,
          description: 'Width of the likert scales in pixels.'
        },
        
      
         //TRIAL PARAMETERS
        trial_duration: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Trial duration',
            default: null,
            description: 'How long to show the trial.'
        }, 
        button_label: {
          type: jsPsych.plugins.parameterType.STRING,
          pretty_name: 'Button label',
          default:  'Continue',
          description: 'Label of the button.'
        }
      }         
    }
  
    plugin.trial = function(display_element, trial) {
      var html = ""

         // setup stimulus
        var html = '<div>'
        html += '<video id="jspsych-video-keyboard-response-stimulus"';

        if(trial.width) {
        html += ' width="'+trial.width+'"';
        }
        if(trial.height) {
        html += ' height="'+trial.height+'"';
        }
        if(trial.autoplay){
        html += " autoplay ";
        }
        if(trial.controls){
        html +=" controls ";
        }
        html +=">";

        var video_preload_blob = jsPsych.pluginAPI.getVideoBuffer(trial.sources[0]);
        if(!video_preload_blob) {
        for(var i=0; i<trial.sources.length; i++){
            var file_name = trial.sources[i];
            if(file_name.indexOf('?') > -1){
            file_name = file_name.substring(0, file_name.indexOf('?'));
            }
            var type = file_name.substr(file_name.lastIndexOf('.') + 1);
            type = type.toLowerCase();
            html+='<source src="' + file_name + '" type="video/'+type+'">';   
         }
        }
        html += "</video>";
        html += "</div>";

       

        if (trial.prompt !== null && trial.prompt_location == "above") {
            html += trial.prompt;
          }
      
          
  
      
          html += '<div '+
          'id="jspsych-free-sort-arena" '+
          'class="jspsych-free-sort-arena" '+
          'style="position: relative; width:'+trial.sort_area_width+'px; height:'+trial.sort_area_height+'px; border:2px solid #444; background-image: url("static/img/scale.png");"' + 
          
          '</div></div>';
    
        // check if prompt exists and if it is shown below
        if (trial.prompt !== null && trial.prompt_location == "below") {
          html += trial.prompt;
        }
    
        display_element.innerHTML = html;
    
        // store initial location data
        var init_locations = [];
    
        for (var i = 0; i < trial.stimuli.length; i++) {
          var coords = random_coordinate(trial.sort_area_width - trial.stim_width, trial.sort_area_height - trial.stim_height);
    
          display_element.querySelector("#jspsych-free-sort-arena").innerHTML += '<img '+
            'src="'+trial.stimuli[i]+'" '+
            'data-src="'+trial.stimuli[i]+'" '+
            'class="jspsych-free-sort-draggable" '+
            'draggable="false" '+
            'style="position: absolute; cursor: move; width:'+trial.stim_width+'px; height:'+trial.stim_height+'px; top:'+coords.y+'px; left:'+coords.x+'px;">'+
            '</img>';
    
          init_locations.push({
            "src": trial.stimuli[i],
            "x": coords.x,
            "y": coords.y
          });
        }

    
        var maxz = 1;
    
        var moves = [];
    
        var draggables = display_element.querySelectorAll('.jspsych-free-sort-draggable');
    
        for(var i=0;i<draggables.length; i++){
          draggables[i].addEventListener('mousedown', function(event){
            var x = event.pageX - event.currentTarget.offsetLeft;
            var y = event.pageY - event.currentTarget.offsetTop - window.scrollY;
            var elem = event.currentTarget;
            elem.style.zIndex = ++maxz;
    
            var mousemoveevent = function(e){
              elem.style.top =  Math.min(trial.sort_area_height - trial.stim_height, Math.max(0,(e.clientY - y))) + 'px';
              elem.style.left = Math.min(trial.sort_area_width  - trial.stim_width,  Math.max(0,(e.clientX - x))) + 'px';
            }
            document.addEventListener('mousemove', mousemoveevent);
    
            var mouseupevent = function(e){
              document.removeEventListener('mousemove', mousemoveevent);
              moves.push({
                "src": elem.dataset.src,
                "x": elem.offsetLeft,
                "y": elem.offsetTop
              });
              document.removeEventListener('mouseup', mouseupevent);
            }
            document.addEventListener('mouseup', mouseupevent);
          });
        }

        //  if(trial.scale_width !== null){
        //     var w = trial.scale_width + 'px';
        // } else {
        //     var w = '100%';
        // }

          // inject CSS for trial
          html += '<style id="jspsych-survey-likert-css">';
          html += ".jspsych-survey-likert-statement { display:block; font-size: 16px; padding-top: 40px; margin-bottom:10px; }"+
            ".jspsych-survey-likert-opts { list-style:none; width:"+w+"; margin:auto; padding:0 0 35px; display:block; font-size: 14px; line-height:1.1em; }"+
            ".jspsych-survey-likert-opt-label { line-height: 1.1em; color: #444; }"+
            ".jspsych-survey-likert-opts:before { content: ''; position:relative; top:11px; /*left:9.5%;*/ display:block; background-color:#efefef; height:4px; width:100%; }"+
            ".jspsych-survey-likert-opts:last-of-type { border-bottom: 0; }"+
            ".jspsych-survey-likert-opts li { display:inline-block; /*width:19%;*/ text-align:center; vertical-align: top; }"+
            ".jspsych-survey-likert-opts li input[type=radio] { display:block; position:relative; top:0; left:50%; margin-left:-6px; }"
          html += '</style>';
      
         
          html += '<form id="jspsych-survey-likert-form">';
      
          // add likert scale questions ///
          // generate question order. this is randomized here as opposed to randomizing the order of trial.questions
          // so that the data are always associated with the same question regardless of order
          var question_order = [];
          for(var i=0; i<trial.questions.length; i++){
            question_order.push(i);
          }
          if(trial.randomize_question_order){
            question_order = jsPsych.randomization.shuffle(question_order);
          }
          
          for (var i = 0; i < trial.questions.length; i++) {
            var question = trial.questions[question_order[i]];
            // add question
            html += '<label class="jspsych-survey-likert-statement">' + question.prompt + '</label>';
            // add options
            var width = 100 / question.labels.length;
            var options_string = '<ul class="jspsych-survey-likert-opts" data-name="'+question.name+'" data-radio-group="Q' + question_order[i] + '">';
            for (var j = 0; j < question.labels.length; j++) {
              options_string += '<li style="width:' + width + '%"><input type="radio" name="Q' + question_order[i] + '" value="' + j + '"';
              if(question.required){
                options_string += ' required';
              }
              options_string += '><label class="jspsych-survey-likert-opt-label">' + question.labels[j] + '</label></li>';
            }
            options_string += '</ul>';
            html += options_string;
          }
      
          // add submit button
          html += '<input type="submit" id="jspsych-survey-likert-next" class="jspsych-survey-likert jspsych-btn" value="'+trial.button_label+'"></input>';
      
          html += '</form>'
      
          display_element.innerHTML = html;
      
          display_element.querySelector('#jspsych-survey-likert-form').addEventListener('submit', function(e){
            e.preventDefault();
            // measure response time
            var endTime = performance.now();
            var response_time = endTime - startTime;
      
            // create object to hold responses
            var question_data = {};
            var matches = display_element.querySelectorAll('#jspsych-survey-likert-form .jspsych-survey-likert-opts');
            for(var index = 0; index < matches.length; index++){
              var id = matches[index].dataset['radioGroup'];
              var el = display_element.querySelector('input[name="' + id + '"]:checked');
              if (el === null) {
                var response = "";
              } else {
                var response = parseInt(el.value);
              }
              var obje = {};
              if(matches[index].attributes['data-name'].value !== ''){
                var name = matches[index].attributes['data-name'].value;
              } else {
                var name = id;
              }
              obje[name] = response;
              Object.assign(question_data, obje);
            }
      
            // save data
            var trial_data = {
              "rt": response_time,
              "responses": JSON.stringify(question_data),
              "question_order": JSON.stringify(question_order)
            };
      
            display_element.innerHTML = '';
      
            // next trial
            jsPsych.finishTrial(trial_data);
          });
      
          var startTime = performance.now();
        };

        function random_coordinate(max_width, max_height) {
            var rnd_x = Math.floor(Math.random() * (max_width - 1));
            var rnd_y = Math.floor(Math.random() * (max_height - 1));
        
            return {
              x: rnd_x,
              y: rnd_y
            };
          }
      
        return plugin;
      })();

      
  