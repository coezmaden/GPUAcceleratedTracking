@inline function eval_results(
    df::DataFrame
)
    simple_data = df  |> @map({_.processor, _.algorithm, _.num_samples, _.num_ants, _.num_correlators, _.TrialObj}) |> @orderby_descending({_.num_samples}) |> DataFrame
    df_pretty = pretty_table(simple_data)
end