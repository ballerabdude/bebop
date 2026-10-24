import { Suspense } from "react";
import { RouterProvider } from "react-router";

import { PageState } from "./components/ui";
import { router } from "./app/router";
import "./App.css";

function App() {
  return (
    <Suspense fallback={<PageState kind="loading" />}>
      <RouterProvider router={router} />
    </Suspense>
  );
}

export default App;
