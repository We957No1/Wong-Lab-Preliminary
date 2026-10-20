import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const base=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const html=fs.readFileSync(path.join(base,'aberration_explorer.html'),'utf8');
const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
const code=scripts[0][1];const context=vm.createContext({});new vm.Script(code).runInContext(context);
const result=vm.runInContext(String.raw`(()=>{
let maxDecomp=0,maxGrad=0;const c=.17,h=.83;
function z(n,m,u,v){return zval(n,m,u,v)}
function decomposition(id,u,v){switch(id){
case'piston':return z(0,0,u,v);case'tilt':return z(1,1,u,v);case'defocus':return z(2,0,u,v);
case'spherical_raw':return z(0,0,u,v)/3+z(2,0,u,v)/(2*Math.sqrt(3))+z(4,0,u,v)/(6*Math.sqrt(5));
case'coma_raw':return h*(z(1,1,u,v)/3+z(3,1,u,v)/(3*Math.sqrt(8)));
case'astig_raw':return h*h*(z(0,0,u,v)/4+z(2,0,u,v)/(4*Math.sqrt(3))+z(2,2,u,v)/(2*Math.sqrt(6)));
case'field_curvature':return h*h*(z(0,0,u,v)/2+z(2,0,u,v)/(2*Math.sqrt(3)));
case'distortion':return h*h*h*z(1,1,u,v)/2;
case'astig':return z(2,2,u,v);case'coma':return z(3,1,u,v);case'spherical':return z(4,0,u,v);case'trefoil':return z(3,3,u,v);case'secondary_spherical':return z(6,0,u,v)}}
for(const mode of MODES){if(!mode.polar.startsWith('W =')||!mode.z.startsWith('W ='))throw Error('Missing formulas');for(let k=1;k<100;k++){const r=Math.sqrt(k/110),th=k*2.399963229728653,u=r*Math.cos(th),v=r*Math.sin(th),b=base(mode.id,u,v,h),eps=1e-6;maxDecomp=Math.max(maxDecomp,Math.abs(b[0]-decomposition(mode.id,u,v)));const gx=(base(mode.id,u+eps,v,h)[0]-base(mode.id,u-eps,v,h)[0])/(2*eps),gy=(base(mode.id,u,v+eps,h)[0]-base(mode.id,u,v-eps,h)[0])/(2*eps);maxGrad=Math.max(maxGrad,Math.abs(gx-b[1]),Math.abs(gy-b[2]))}}
const ideal=compute('piston',0,0,1),piston=compute('piston',.35,0,1),tilt=compute('tilt',.5,0,1),N=ideal.N;let pistonErr=0,tiltErr=0,maxEnergy=0;const shift=N/ideal.R;
for(let y=0;y<N;y++)for(let x=0;x<N;x++){pistonErr=Math.max(pistonErr,Math.abs(ideal.I[y*N+x]-piston.I[y*N+x]));tiltErr=Math.max(tiltErr,Math.abs(tilt.I[y*N+x]-ideal.I[y*N+(x-shift+N)%N]))}
for(const mode of MODES){const res=compute(mode.id,.13,.04,.9);if(!Number.isFinite(res.peak)||res.peak>1+1e-10)throw Error('Invalid peak '+mode.id);maxEnergy=Math.max(maxEnergy,Math.abs(res.energy/ideal.energy-1))}
if(maxDecomp>1e-12||maxGrad>1e-7||pistonErr>1e-12||tiltErr>1e-12||maxEnergy>1e-11)throw Error('Numerical checks failed');
return{modes:MODES.length,max_decomposition_error:maxDecomp,max_gradient_finite_difference_error:maxGrad,piston_intensity_error:pistonErr,positive_tilt_shift_error:tiltErr,phase_only_energy_error:maxEnergy};
})()`,context,{timeout:60000});
const original=fs.readFileSync(path.join(base,'checks/original_visualization_payload.html'),'utf8');
const adapted=fs.readFileSync(path.join(base,'aberration_illustration.html'),'utf8');
const extract=s=>JSON.parse(s.match(/<script type="application\/json" id="pupil-demo-data">([\s\S]*?)<\/script>/)[1]);
if(JSON.stringify(extract(original))!==JSON.stringify(extract(adapted)))throw Error('Original data changed');
for(const name of ['aberration_illustration.html','aberration_explorer.html']){
const text=fs.readFileSync(path.join(base,name),'utf8');
if(/<script[^>]*src=|<link[^>]*href=|@import\s|url\(https?:/i.test(text))throw Error('External runtime dependency');
for(const script of text.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g))if(!script[1].includes('application/json'))new vm.Script(script[2]);
}
result.original_7_case_data_preserved=true;result.no_external_runtime_dependencies=true;result.browser_visual_review='Not available: CUA reported no browsers. Numerical, syntax, and offline-dependency checks completed.';result.status='PASS';
fs.writeFileSync(path.join(base,'checks/html_verification.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
