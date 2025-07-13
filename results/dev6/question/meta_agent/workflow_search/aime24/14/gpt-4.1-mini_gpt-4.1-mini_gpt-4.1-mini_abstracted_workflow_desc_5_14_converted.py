async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: aggregate_valid_combinations_and_derive_composite_measure

    # Sub-task 1: Formulate geometric and algebraic conditions
    cot_instruction_0_1 = (
        "Sub-task 1: Formulate the geometric and algebraic conditions defining the rhombi inscribed in the hyperbola "
        "x^2/20 - y^2/24 = 1 with diagonals intersecting at the origin. Express points A, B, C, D as vectors such that A = -C, B = -D, "
        "and all points lie on the hyperbola. Identify parametric forms of points on the hyperbola to facilitate algebraic manipulation. "
        "Avoid assuming any particular orientation beyond midpoint conditions."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formulating geometric and algebraic conditions, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Sub-task 2: Derive algebraic conditions from rhombus properties
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the formulation from Sub-task 1, derive algebraic conditions imposed by the rhombus properties: equal side lengths and perpendicular diagonals. "
        "Translate these into equations involving coordinates or parameters of A and B. Emphasize perpendicularity of diagonals (dot product zero) and equality of side lengths (distance formulas). "
        "Avoid premature simplifications that might exclude valid solutions."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, deriving algebraic conditions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)

    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo] + possible_answers_0_2 + possible_thinkings_0_2,
        "Sub-task 2: Synthesize and choose the most consistent algebraic conditions for the rhombus inscribed in the hyperbola.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, synthesizing algebraic conditions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Sub-task 3: Combine hyperbola and rhombus conditions to characterize valid rhombi
    cot_instruction_0_3 = (
        "Sub-task 3: Combine the hyperbola constraints and rhombus conditions from Sub-tasks 1 and 2 to obtain a system of equations characterizing all valid rhombi with diagonals intersecting at the origin. "
        "Express the squared length of diagonal BD in terms of the parameters introduced. Prepare the system for parameter elimination or substitution in the next stage."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3(
        [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2],
        cot_instruction_0_3,
        is_sub_task=True
    )
    agents.append(f"CoT agent {cot_agent_0_3.id}, combining conditions and expressing BD^2, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Stage 1: infer_compute_parameters_from_composite_data

    # Sub-task 1: Solve system for parameters and express BD^2
    cot_instruction_1_1 = (
        "Sub-task 1: Analyze the system of equations from Stage 0 to solve for the parameters describing the rhombi. "
        "Use algebraic manipulation and substitution to express the squared length of diagonal BD as a function of a single parameter or reduced parameters. "
        "Consider domain restrictions from the hyperbola and rhombus conditions. Avoid ignoring boundary or limiting cases that might affect the supremum."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, solving system for parameters and BD^2, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Sub-task 2: Investigate behavior of BD^2 function to find supremum
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Investigate the behavior of the squared diagonal length function derived in Sub-task 1 to identify its maximum or supremum value. "
        "Use calculus (critical points), inequality analysis, or limit evaluation. Confirm if supremum is attainable or characterize as a limit. Avoid assuming maximum is attained without verification."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, investigating BD^2 behavior, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + possible_answers_1_2 + possible_thinkings_1_2,
        "Sub-task 2: Synthesize and choose the most consistent supremum value for BD^2.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, synthesizing supremum of BD^2, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Stage 2: aggregate_and_combine_values

    # Sub-task 1: Aggregate results and state supremum explicitly
    cot_instruction_2_1 = (
        "Sub-task 1: Aggregate the results from Stage 1 to explicitly state the greatest real number less than BD^2 for all such rhombi. "
        "Provide the final numeric or symbolic expression for this supremum. Verify consistency with problem constraints and interpret geometric meaning. "
        "Avoid leaving the answer implicit or incomplete."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, aggregating and stating supremum, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Sub-task 2: Verification by sample rhombi near extremal case
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Perform verification by checking the derived supremum against sample rhombi constructed from parameter values close to the extremal case. "
        "Confirm BD^2 values approach but do not exceed the supremum. Synthesize verification with final answer to ensure correctness and completeness."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying supremum, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_2_2(
            [taskInfo, thinking_2_2, answer_2_2],
            "Please review and provide limitations of the verification. If absolutely correct, output exactly 'True' in 'correct'.",
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs