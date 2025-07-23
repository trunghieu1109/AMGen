async def forward_158(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []
    results = {}

    for _ in range(1):
        cot_instruction0_0 = "Sub-task 0: Extract and summarize the given observational and cosmological information relevant to the quasar and the problem."
        cot_agent_desc0_0 = {
            "instruction": cot_instruction0_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results0_0, log0_0 = await self.cot(
            subtask_id="stage_0.subtask_0",
            cot_agent_desc=cot_agent_desc0_0
        )
        logs.append(log0_0)

        cot_instruction0_1 = (
            "Sub-task 1: Estimate the quasar's redshift z by relating the observed peak wavelength (790 nm) "
            "to an assumed intrinsic emission wavelength, using the output from Sub-task 0."
        )
        cot_agent_desc0_1 = {
            "instruction": cot_instruction0_1,
            "input": [taskInfo, results0_0["thinking"], results0_0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
        }
        results0_1, log0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc0_1
        )
        logs.append(log0_1)

        cot_instruction0_2 = (
            "Sub-task 2: Calculate the comoving distance corresponding to the estimated redshift "
            "using the Lambda-CDM cosmological parameters (H0=70, Omega_m=0.3, Omega_Lambda=0.7, flat universe), "
            "based on the output from Sub-task 1."
        )
        cot_agent_desc0_2 = {
            "instruction": cot_instruction0_2,
            "input": [taskInfo, results0_1["thinking"], results0_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results0_2, log0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc0_2
        )
        logs.append(log0_2)

        aggregate_instruction0_3 = (
            "Sub-task 3: Refine and consolidate the intermediate results from previous subtasks, "
            "simplifying and documenting the reasoning and calculations to produce a clear provisional comoving distance estimate."
        )
        aggregate_desc0_3 = {
            "instruction": aggregate_instruction0_3,
            "input": [taskInfo, results0_0, results0_1, results0_2],
            "temperature": 0.0,
            "context": ["user query", "solutions from stage_0.subtask_0", "solutions from stage_0.subtask_1", "solutions from stage_0.subtask_2"]
        }
        results0_3, log0_3 = await self.aggregate(
            subtask_id="stage_0.subtask_3",
            aggregate_desc=aggregate_desc0_3
        )
        logs.append(log0_3)

        results["stage_0"] = {
            "subtask_0": results0_0,
            "subtask_1": results0_1,
            "subtask_2": results0_2,
            "subtask_3": results0_3
        }

    cot_agent_instruction1_0 = (
        "Sub-task 0: Evaluate the provisional comoving distance estimate against the provided multiple-choice options (6, 7, 8, 9 Gpc) "
        "and select the best matching candidate, based on the output from stage_0.subtask_3."
    )
    cot_agent_desc1_0 = {
        "instruction": cot_agent_instruction1_0,
        "input": [taskInfo, results["stage_0"]["subtask_3"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_3"]
    }
    results1_0, log1_0 = await self.answer_generate(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc1_0
    )
    logs.append(log1_0)

    final_answer = await self.make_final_answer(results1_0["thinking"], results1_0["answer"])
    return final_answer, logs
