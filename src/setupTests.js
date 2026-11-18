// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';
import {TextEncoder,TextDecoder} from 'util';
import {webcrypto} from 'crypto';
import {serialize,deserialize} from 'v8';
Object.defineProperty(global,'crypto',{value:webcrypto,configurable:true});
global.TextEncoder=TextEncoder;global.TextDecoder=TextDecoder;
global.structuredClone=value=>deserialize(serialize(value));
