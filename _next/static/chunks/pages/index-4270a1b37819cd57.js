(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[405],{8312:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/",function(){return n(2603)}])},3740:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default=void 0;var i=n(6495).Z,o=n(2648).Z,r=n(1598).Z,a=n(7273).Z,s=r(n(7294)),l=o(n(2636)),c=n(7757),d=n(3735),u=n(3341);n(4210);var f=o(n(7746));let g={deviceSizes:[640,750,828,1080,1200,1920,2048,3840],imageSizes:[16,32,48,64,96,128,256,384],path:"/_next/image",loader:"default",dangerouslyAllowSVG:!1,unoptimized:!0};function p(e){return void 0!==e.default}function h(e){return"number"==typeof e||void 0===e?e:"string"==typeof e&&/^[0-9]+$/.test(e)?parseInt(e,10):NaN}function m(e,t,n,o,r,a,s){if(!e||e["data-loaded-src"]===t)return;e["data-loaded-src"]=t;let l="decode"in e?e.decode():Promise.resolve();l.catch(()=>{}).then(()=>{if(e.parentElement&&e.isConnected){if("blur"===n&&a(!0),null==o?void 0:o.current){let t=new Event("load");Object.defineProperty(t,"target",{writable:!1,value:e});let n=!1,r=!1;o.current(i({},t,{nativeEvent:t,currentTarget:e,target:e,isDefaultPrevented:()=>n,isPropagationStopped:()=>r,persist:()=>{},preventDefault:()=>{n=!0,t.preventDefault()},stopPropagation:()=>{r=!0,t.stopPropagation()}}))}(null==r?void 0:r.current)&&r.current(e)}})}let w=s.forwardRef((e,t)=>{var{imgAttributes:n,heightInt:o,widthInt:r,qualityInt:l,className:c,imgStyle:d,blurStyle:u,isLazy:f,fill:g,placeholder:p,loading:h,srcString:w,config:v,unoptimized:b,loader:y,onLoadRef:_,onLoadingCompleteRef:j,setBlurComplete:x,setShowAltText:E,onLoad:S,onError:C}=e,k=a(e,["imgAttributes","heightInt","widthInt","qualityInt","className","imgStyle","blurStyle","isLazy","fill","placeholder","loading","srcString","config","unoptimized","loader","onLoadRef","onLoadingCompleteRef","setBlurComplete","setShowAltText","onLoad","onError"]);return h=f?"lazy":h,s.default.createElement(s.default.Fragment,null,s.default.createElement("img",Object.assign({},k,{loading:h,width:r,height:o,decoding:"async","data-nimg":g?"fill":"1",className:c,style:i({},d,u)},n,{ref:s.useCallback(e=>{t&&("function"==typeof t?t(e):"object"==typeof t&&(t.current=e)),e&&(C&&(e.src=e.src),e.complete&&m(e,w,p,_,j,x,b))},[w,p,_,j,x,C,b,t]),onLoad:e=>{let t=e.currentTarget;m(t,w,p,_,j,x,b)},onError:e=>{E(!0),"blur"===p&&x(!0),C&&C(e)}})))}),v=s.forwardRef((e,t)=>{let n,o;var r,{src:m,sizes:v,unoptimized:b=!1,priority:y=!1,loading:_,className:j,quality:x,width:E,height:S,fill:C,style:k,onLoad:z,onLoadingComplete:N,placeholder:P="empty",blurDataURL:A,layout:R,objectFit:M,objectPosition:O,lazyBoundary:I,lazyRoot:L}=e,B=a(e,["src","sizes","unoptimized","priority","loading","className","quality","width","height","fill","style","onLoad","onLoadingComplete","placeholder","blurDataURL","layout","objectFit","objectPosition","lazyBoundary","lazyRoot"]);let D=s.useContext(u.ImageConfigContext),T=s.useMemo(()=>{let e=g||D||d.imageConfigDefault,t=[...e.deviceSizes,...e.imageSizes].sort((e,t)=>e-t),n=e.deviceSizes.sort((e,t)=>e-t);return i({},e,{allSizes:t,deviceSizes:n})},[D]),F=B,W=F.loader||f.default;delete F.loader;let H="__next_img_default"in W;if(H){if("custom"===T.loader)throw Error('Image with src "'.concat(m,'" is missing "loader" prop.')+"\nRead more: https://nextjs.org/docs/messages/next-image-missing-loader")}else{let e=W;W=t=>{let{config:n}=t,i=a(t,["config"]);return e(i)}}if(R){"fill"===R&&(C=!0);let e={intrinsic:{maxWidth:"100%",height:"auto"},responsive:{width:"100%",height:"auto"}}[R];e&&(k=i({},k,e));let t={responsive:"100vw",fill:"100vw"}[R];t&&!v&&(v=t)}let U="",V=h(E),q=h(S);if("object"==typeof(r=m)&&(p(r)||void 0!==r.src)){let e=p(m)?m.default:m;if(!e.src)throw Error("An object should only be passed to the image component src parameter if it comes from a static image import. It must include src. Received ".concat(JSON.stringify(e)));if(!e.height||!e.width)throw Error("An object should only be passed to the image component src parameter if it comes from a static image import. It must include height and width. Received ".concat(JSON.stringify(e)));if(n=e.blurWidth,o=e.blurHeight,A=A||e.blurDataURL,U=e.src,!C){if(V||q){if(V&&!q){let t=V/e.width;q=Math.round(e.height*t)}else if(!V&&q){let t=q/e.height;V=Math.round(e.width*t)}}else V=e.width,q=e.height}}let G=!y&&("lazy"===_||void 0===_);((m="string"==typeof m?m:U).startsWith("data:")||m.startsWith("blob:"))&&(b=!0,G=!1),T.unoptimized&&(b=!0),H&&m.endsWith(".svg")&&!T.dangerouslyAllowSVG&&(b=!0);let[Z,J]=s.useState(!1),[X,Y]=s.useState(!1),$=h(x),K=Object.assign(C?{position:"absolute",height:"100%",width:"100%",left:0,top:0,right:0,bottom:0,objectFit:M,objectPosition:O}:{},X?{}:{color:"transparent"},k),Q="blur"===P&&A&&!Z?{backgroundSize:K.objectFit||"cover",backgroundPosition:K.objectPosition||"50% 50%",backgroundRepeat:"no-repeat",backgroundImage:'url("data:image/svg+xml;charset=utf-8,'.concat(c.getImageBlurSvg({widthInt:V,heightInt:q,blurWidth:n,blurHeight:o,blurDataURL:A,objectFit:K.objectFit}),'")')}:{},ee=function(e){let{config:t,src:n,unoptimized:i,width:o,quality:r,sizes:a,loader:s}=e;if(i)return{src:n,srcSet:void 0,sizes:void 0};let{widths:l,kind:c}=function(e,t,n){let{deviceSizes:i,allSizes:o}=e;if(n){let e=/(^|\s)(1?\d?\d)vw/g,t=[];for(let i;i=e.exec(n);i)t.push(parseInt(i[2]));if(t.length){let e=.01*Math.min(...t);return{widths:o.filter(t=>t>=i[0]*e),kind:"w"}}return{widths:o,kind:"w"}}if("number"!=typeof t)return{widths:i,kind:"w"};let r=[...new Set([t,2*t].map(e=>o.find(t=>t>=e)||o[o.length-1]))];return{widths:r,kind:"x"}}(t,o,a),d=l.length-1;return{sizes:a||"w"!==c?a:"100vw",srcSet:l.map((e,i)=>"".concat(s({config:t,src:n,quality:r,width:e})," ").concat("w"===c?e:i+1).concat(c)).join(", "),src:s({config:t,src:n,quality:r,width:l[d]})}}({config:T,src:m,unoptimized:b,width:V,quality:$,sizes:v,loader:W}),et=m,en={imageSrcSet:ee.srcSet,imageSizes:ee.sizes,crossOrigin:F.crossOrigin},ei=s.useRef(z);s.useEffect(()=>{ei.current=z},[z]);let eo=s.useRef(N);s.useEffect(()=>{eo.current=N},[N]);let er=i({isLazy:G,imgAttributes:ee,heightInt:q,widthInt:V,qualityInt:$,className:j,imgStyle:K,blurStyle:Q,loading:_,config:T,fill:C,unoptimized:b,placeholder:P,loader:W,srcString:et,onLoadRef:ei,onLoadingCompleteRef:eo,setBlurComplete:J,setShowAltText:Y},F);return s.default.createElement(s.default.Fragment,null,s.default.createElement(w,Object.assign({},er,{ref:t})),y?s.default.createElement(l.default,null,s.default.createElement("link",Object.assign({key:"__nimg-"+ee.src+ee.srcSet+ee.sizes,rel:"preload",as:"image",href:ee.srcSet?void 0:ee.src},en))):null)});t.default=v,("function"==typeof t.default||"object"==typeof t.default&&null!==t.default)&&void 0===t.default.__esModule&&(Object.defineProperty(t.default,"__esModule",{value:!0}),Object.assign(t.default,t),e.exports=t.default)},7757:function(e,t){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.getImageBlurSvg=function(e){let{widthInt:t,heightInt:n,blurWidth:i,blurHeight:o,blurDataURL:r,objectFit:a}=e,s=i||t,l=o||n,c=r.startsWith("data:image/jpeg")?"%3CfeComponentTransfer%3E%3CfeFuncA type='discrete' tableValues='1 1'/%3E%3C/feComponentTransfer%3E%":"";return s&&l?"%3Csvg xmlns='http%3A//www.w3.org/2000/svg' viewBox='0 0 ".concat(s," ").concat(l,"'%3E%3Cfilter id='b' color-interpolation-filters='sRGB'%3E%3CfeGaussianBlur stdDeviation='").concat(i&&o?"1":"20","'/%3E").concat(c,"%3C/filter%3E%3Cimage preserveAspectRatio='none' filter='url(%23b)' x='0' y='0' height='100%25' width='100%25' href='").concat(r,"'/%3E%3C/svg%3E"):"%3Csvg xmlns='http%3A//www.w3.org/2000/svg'%3E%3Cimage style='filter:blur(20px)' preserveAspectRatio='".concat("contain"===a?"xMidYMid":"cover"===a?"xMidYMid slice":"none","' x='0' y='0' height='100%25' width='100%25' href='").concat(r,"'/%3E%3C/svg%3E")}},7746:function(e,t){"use strict";function n(e){let{config:t,src:n,width:i,quality:o}=e;return"".concat(t.path,"?url=").concat(encodeURIComponent(n),"&w=").concat(i,"&q=").concat(o||75)}Object.defineProperty(t,"__esModule",{value:!0}),t.default=void 0,n.__next_img_default=!0,t.default=n},2603:function(e,t,n){"use strict";n.r(t),n.d(t,{default:function(){return f}});var i=n(5893),o=n(7294),r=n(9394),a=n(5675),s=n.n(a),l=n(9034),c=n.n(l);n(3159),n(6196);var d=n(9008),u=n.n(d);function f(){let[e,t]=(0,o.useState)(0),[n,a]=(0,o.useState)(0),[l,d]=(0,o.useState)(void 0);return(0,o.useEffect)(()=>{a(window.innerWidth),window.addEventListener("resize",()=>{a(window.innerWidth)}),d(navigator.pdfViewerEnabled)},[]),(0,i.jsxs)(i.Fragment,{children:[(0,i.jsxs)(u(),{children:[(0,i.jsx)("title",{children:"T3000 - Analyse von Audiosignalen unter der Verwendung von Linear Predictive Coding"}),(0,i.jsx)("meta",{name:"description",content:"Projektarbeit T3000 von Henry Schuler"}),(0,i.jsx)("meta",{name:"viewport",content:"width=device-width, initial-scale=1"}),(0,i.jsx)("link",{rel:"icon",href:"/favicon.ico"})]}),(0,i.jsx)("main",{children:l?(0,i.jsx)("div",{style:{position:"absolute",left:"0",right:"0",bottom:"0",top:"0",overflow:"hidden"},children:(0,i.jsx)("iframe",{src:"./main.pdf",width:"100%",height:"100%",frameBorder:0})}):(0,i.jsxs)("div",{className:c().wrapper,children:[(0,i.jsx)(r.BB,{className:c().document,file:"main.pdf",onLoadSuccess:function(e){let{numPages:n}=e;t(n)},options:{cMapUrl:"https://unpkg.com/pdfjs-dist@".concat(r.v0.version,"/cmaps/"),cMapPacked:!0,standardFontDataUrl:"https://unpkg.com/pdfjs-dist@".concat(r.v0.version,"/standard_fonts")},children:[...Array(e)].map((e,t)=>(0,i.jsx)(r.T3,{width:n,pageNumber:t+1,className:c().page},"pdf-page-"+t))}),(0,i.jsx)("button",{className:c().downloadButton,onClick:()=>{window.open("./main.pdf","_blank")},children:(0,i.jsx)(s(),{src:"/download-install-line-icon.svg",alt:"Download Logo",className:c().vercelLogo,width:20,height:20,priority:!0})})]})})]})}},3159:function(){},6196:function(){},9034:function(e){e.exports={wrapper:"Home_wrapper__kA9A_",document:"Home_document__DMIJk",page:"Home_page__0ydta",downloadButton:"Home_downloadButton__E5rjA"}},9008:function(e,t,n){e.exports=n(2636)},5675:function(e,t,n){e.exports=n(3740)}},function(e){e.O(0,[774,888,179],function(){return e(e.s=8312)}),_N_E=e.O()}]);