import { createStore, combineReducers, applyMiddleware, compose } from 'redux';
import { taskMiddleware } from 'react-palm/tasks';
import { keplerGlReducer } from '@kepler.gl/reducers';

const reducers = combineReducers({
  keplerGl: keplerGlReducer
});

const enhancers = compose(applyMiddleware(taskMiddleware));

const store = createStore(reducers, {}, enhancers);

export default store;
