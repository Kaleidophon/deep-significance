function stickySidebar() {
    
    var topPadding = $('.navbar').outerHeight(true);
    
    var $sidebar = $('.sphinxsidebar');
    var $sidebarwrapper = $('.sphinxsidebarwrapper');
    var $footer = $('footer');
    var $window = $(window);

    var sidebarTop = $sidebar.offset().top;
    var footerTop = $footer.offset().top

    function adjust() {
        var sidebarWidth = $sidebar.width();
        var sidebarWrapperHeight = $sidebarwrapper.height();
        var scrollTop = $window.scrollTop();

        if (scrollTop + topPadding <= sidebarTop) {
          $sidebarwrapper.css({'top': 0, 'position': 'relative' });
        } else {
          $sidebarwrapper.css({'top': topPadding, 'position': 'fixed', 'width': sidebarWidth });
        }

        $sidebarwrapper.css({ 'max-height': footerTop - scrollTop - topPadding, 'overflow': 'auto' });
    };

    var scrollTimer;
    $window.scroll(function() {
        if (scrollTimer) { clearTimeout(scrollTimer); }

        scrollTimer = setTimeout(function() {
            adjust();
        }, 1);
    });

    var resizeTimer;
    $window.on('resize', function() {
        if (resizeTimer) { clearTimeout(resizeTimer); }

        resizeTimer = setTimeout(function() {
            adjust();
        }, 1);
    });

    adjust();
}

function sidebarTreeView() {
    var $toctree = $('.sidebartoctree > ul');

    $toctree.find('li').each(function(index, element) {
        $li = $(element).has('ul')

        $li.children('a').append('<a class="arrow collapsed" data-toggle="collapse" href="#collapse' + index + '" aria-expanded="true"></a>').end()
            .children('ul').addClass('collapse').attr({'id': 'collapse' + index}).end();
        if ($li.hasClass('current')) {
            $li.children('ul').addClass('show').end()
                .children('a').children('a').removeClass('collapsed');
        }
    });

}

$(function() {
    $('body').fadeIn(0).scrollspy({ target: 'li.current' });

    // Grid layout Style
    $(".sphinxsidebarwrapper > .sidebartoctree > ul").addClass('nav flex-column nav-pills')
        .find('li').addClass('nav-item').end()
        .find('a.reference').addClass('nav-link').end()

    
    $(".sphinxsidebar").addClass("col-md-3");
    $(".document").addClass("col-md-9");
    sidebarTreeView();
    
    $(".related").addClass("col-md-12");
    $(".footer").addClass("col-md-12");

    
    // Navbar Globaltoc Style
    $("#navbar-pages li.toctree-l1").unwrap();
    $('#navbar-pages > ul').children('li')
        .find('a').addClass('dropdown-item').end()
        .has('ul').addClass('dropdown')
            .find('ul').addClass('dropdown-menu');
    $("#navbar-pages > ul").find("li").has("ul").children("a").addClass("arrow");// Tables
    $("table.docutils").addClass("table table-sm table-bordered table-striped")
        .find("thead")
        
        .addClass("thead-inverse")
        

    // Admonition
    $(".admonition").addClass("alert")
        .filter(".hint").addClass("alert-info").children('p.admonition-title').prepend('<div class="icon"><div class="question-mark"></div></div>').end().end()
        .filter(".note, .warning").addClass("alert-warning").children('p.admonition-title').prepend('<div class="icon"><div class="information-mark"></div></div>').end().end()
        .filter(".tip, .important").addClass("alert-success").children('p.admonition-title').prepend('<div class="icon"><div class="check-mark"></div></div>').end().end()
        .filter(".caution, .danger, .error").addClass("alert-danger").children('p.admonition-title').prepend('<div class="icon"><div class="exclamation-mark"></div></div>').end().end();

    // images
    $("img").addClass("img-fluid");

    // download
    $("a.download").prepend('<div class="icon"><div class="download"></div></div>');

    
    stickySidebar();
    

});