
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
        $(this).parents('tr').set_enabled($(this).is(":checked"));
    });
    $('.pe-table .enable-row').trigger('change');

    // define the getter/setters
    $('.setting button').on('click', function() {
        var id = $(this).parents('tr').attr('id');
        var value = $(this).parents('tr').get_value();
        connection.emit('api', 'SET-SETTING', [id, value])
    });
    $('#setting-set-all').on('click', function() {
        $('.setting button').trigger('click');
    });
    $('.setting .indicator').on('click', function() {
        var id = $(this).parents('tr').attr('id');
        connection.emit('api', 'GET-SETTING', [id]);
    });
    $('#setting-get-all').on('click', function() {
        console.log('get-all')
        $('.setting .indicator').trigger('click');
        $('.model-state').trigger('update');
    });

    $('#pipeline select').on('change', function() {
        var url = '/pipeline/' + $(this).get_value();
        window.location = url;
    });

    // define the application action handlers
    $('#connection_status button').on('click', function() {
        connection.toggle();
    });
    $('#output_file button').on('click', function() {
        $('.sequence').each(function(index) {
            var is_enabled = $(this).is_enabled();
            var key = $('select', $(this)).val();
            var range = $('input[type="text"]', $(this)).val();
            connection.emit('api', 'SET-SEQUENCE', [index, key, range, is_enabled])
        });
        var value = $(this).get_value();
        connection.emit('api', 'RUN-PIPELINE', [value]);
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
    $('.model-state').on('update', function(event) {
        connection.emit('api', 'GET-MODEL-STATE', [this.id]);
    });

    connection.init($('#connection_status'), $('#pyxel'))
    connection.open();

    if ($('#pipeline select').get_value() == '') {
        $('#group-0').css('display', 'none');
        $('#group-1').css('display', 'none');
        $('#group-2').css('display', 'none');
    }

});
