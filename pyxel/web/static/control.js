
$(document).ready(function() {
    viewer = null;

    // set bootstrap styles
    $('.pe-tab').addClass('tab-pane fade in');
    $('.pe-tab:first-child').addClass('active');
    $('.pe-tab:last-child').addClass('active');
    $('.pe-table td:nth-child(2) div').addClass('form-control pe-output')
    $('.pe-table td input[type="text"]').addClass('form-control')
    $('.pe-table td input[type="number"]').addClass('form-control')
    $('.pe-table td:last-child input[type="checkbox"]').addClass('form-control')
    $('.pe-table td select').addClass('form-control')
    $('.pe-table td select').addClass('custom')
    $('.pe-table td button').addClass('btn')
    $('.pe-table td button').addClass('btn-primary')
    $('input[type=number]').each(function(){
        var step = $(this).prop('step');
        var default_decimals = 0;
        if (step.indexOf('.') != -1) {
            default_decimals = step.length - step.indexOf('.');
        }
        console.log('decimals:' + default_decimals)
        $(this).TouchSpin({
            min: $(this).prop('min'),
            max: $(this).prop('max'),
            step: $(this).prop('step'),
            decimals: ($(this).attr('decimals') || default_decimals),
            verticalbuttons: true
        });
    });

    // define the section event handlers
    $('#pe-expand').on('click', function() {
        $('.pe-section-hide').find('caption').trigger('click');
    });
    $('#pe-collapse').on('click', function() {
        $('.pe-table').not('.pe-section-hide').find('caption').trigger('click');
    });
    $('.pe-section').each(function() {
        $(this).prop('title', $(this).text());
        $(this).html('&#x25BC ' + $(this).text());
    });
    $('.pe-section').on('click', function() {
        $(this).closest('table').toggleClass('pe-section-hide');
        var is_hidden = $(this).closest('table').hasClass('pe-section-hide');
        var arrow = is_hidden ? '&nbsp;&#x25B6; ' : '&nbsp;&#x25BC ';
        $(this).html(arrow + $(this).prop('title'));
    });

    // define the row enablement checkboxs
    $('.pe-table .enable-row').on('change', function() {
        $(this).get_context().set_enabled($(this).is(":checked"));
    });
    $('.pe-table .enable-row').trigger('change');

    // define the getter/setters
    $('.setting button').on('click', function() {
        var id = $(this).get_context().attr('id');
        var value = $(this).get_context().get_value();
        connection.emit('api', 'SET-SETTING', [id, value])
    });
    $('#setting-set-all').on('click', function() {
        $('.setting button').trigger('click');
    });
    $('.setting .indicator').on('click', function() {
        var id = $(this).get_context().attr('id');
        connection.emit('api', 'GET-SETTING', [id]);
    });
    $('#setting-get-all').on('click', function() {
        console.log('get-all')
        $('.setting .indicator').trigger('click');
        $('.model-state').trigger('update');
    });
    $('#get-state').on('click', function() {
        console.log('get-state')
        connection.emit('api', 'GET-STATE', [])
    });

    // define the application action handlers
    $('#connection_status button').on('click', function() {
        connection.toggle();
    });
    $('#pipeline select').on('change', function() {
        var url = '/pipeline/' + $(this).get_value();
        window.location = url;
    });
    $('#generate button').on('click', function() {
        connection.emit('api', 'EXECUTE-CALL', ['save_config', $(this).get_value()]);
    });
    $('#load button').on('click', function() {
        connection.emit('api', 'EXECUTE-CALL', ['load_config', $(this).get_value()]);
    });
    $('#load-module button').on('click', function() {
        connection.emit('api', 'EXECUTE-CALL', ['load_modules', $(this).get_value()]);
        $('#pipeline select').trigger('change');
    });
    $('#registry button').on('click', function() {
        connection.emit('api', 'EXECUTE-CALL', ['load_registry', $(this).get_value()]);
        $('#pipeline select').trigger('change');
    });
    $('#update-from-file button').on('click', function() {
        connection.emit('api', 'EXECUTE-CALL', ['load_defaults', $(this).get_value()]);
    });
    $('#sequence-mode button').on('click', function() {
        var run_mode = $('#sequence-mode input[name=mode]:checked').val();
        connection.emit('api', 'EXECUTE-CALL', ['set_sequence_mode', run_mode])
        $('.sequence').each(function(index) {
            var is_enabled = $(this).is_enabled();
            var key = $('select', $(this)).val();
            var range = $('input[type="text"]', $(this)).val();
            connection.emit('api', 'SET-SEQUENCE', [index, key, range, is_enabled])
        });
    });

    $('#state button').on('click', function() {
        var output_file = $('#output_file').get_value();
        connection.emit('api', 'RUN-PIPELINE', [output_file]);
    });

    $('#setting-reset').on('click', function() {
        alert('To be implemented');
    });
    $('#start-imager').on('click', function(event) {
        event.preventDefault();
        var url = '/js9/js9.html';  //$this.attr("href");
        var name = "JS9 PyXEL Viewer";
        var specs = 'width=500,height=500'
        viewer = window.open(url, name, specs);
        viewer.focus()
        // setTimeout(function() {viewer.load_fits('/data/pyxel/test2.fits')}, 2000);
    });

    $('.model-state').on('change', function() {
        connection.emit('api', 'SET-MODEL-STATE', [this.id, this.checked]);
    });

    $('#pyxel').on('message:progress', function(event, selector, fields) {
        $('.pe-table').trigger('message:get', [selector, fields]);
        var label_color = {'pause': 'orange', 'error': 'red', 'aborted': 'orange'};
        $(selector).progress(fields.state > 0, label_color[fields.value]);
        if (fields.state >= 1) {
            $('#run button').text('Stop')
        }
        if (fields.state <= 0) {
            $('#run button').text('Run')
        }
        if (fields.file) {
            window.frames['js9viewer'].load_fits(fields.file);
            if (viewer) {
                try {
                    viewer.load_fits(fields.file)
                } catch(e) {
                    viewer = null;
                }
            }
        }
        $('#output_file button').text(fields.state ? 'Stop': 'Run')
    });
    $('#pyxel').on('message:get', function(event, selector, fields) {
        if (fields.value != null) {
            var text = $.format('%(value)s', fields);
        }
        $(selector).highlight(text);
        if (selector.indexOf('sequence_') == -1) {
            $(selector).set_value(fields.value);
        }
    });
    $('#pyxel').on('message:enabled', function(event, selector, fields) {
        $(selector).prop('checked', fields.value);
    });
    $('#pyxel').on('message:state', function(event, selector, fields) {
        for (var key in fields.value) {
            var selector = key.split('.').slice(1).join('.');
            selector = selector.replace('.models', '');
            selector = '#' + selector.replace(/\./g, '\\.');
            console.log('message:state', selector, fields.value[key])
            if (key.match(/steps\.[0-9]\.enabled/g)) {
                $(selector + ' .enable-row').prop('checked', fields.value[key]);
                $(selector + ' .enable-row').trigger('change');
            } else if (key.match(/pipeline\..*\.enabled/g)) {
                $(selector).prop('checked', fields.value[key]);
            } else if (key == 'parametric.mode') {
                $('input[name=mode]').filter('[value="'+fields.value[key]+'"]').prop('checked', true);
            } else {
                $(selector).highlight(fields.value[key], true);
                $(selector).set_value(fields.value[key], false);
            }
        }
    });
    $('#pyxel').on('message:error', function(event, selector, fields) {
        $(selector + ' .indicator').css('background-color', 'pink')
        $(selector + ' .indicator').text('error')
        $(selector + ' .indicator').prop('title', fields.value)
    });

    $('.model-state').on('update', function(event) {
        connection.emit('api', 'GET-MODEL-STATE', [this.id]);
    });

    if ($('#pipeline select').get_value() == '') {
        $('#group-0').css('display', 'none');
        $('#group-1').css('display', 'none');
        $('#group-2').css('display', 'none');
    }

    connection.init($('#connection_status'), $('#pyxel'), function(){$('#get-state').trigger('click');})
    connection.open();
});