async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Domain and geometric conditions (SC_CoT)

    cot_sc_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state the domain of points A, B, C, and D on the hyperbola defined by "
        "x^2/20 - y^2/24 = 1. Emphasize the symmetry of the hyperbola about both axes and the fact that the origin is its center. "
        "Avoid attempting to solve any geometric constraints at this stage; focus solely on the domain characterization.")
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, domain characterization, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 1: Synthesize and choose the most consistent domain characterization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Formulate the geometric conditions that ABCD is a rhombus with diagonals intersecting at the origin. "
        "Explicitly state that the origin is the midpoint of both diagonals and that the diagonals are perpendicular. "
        "Emphasize the implications of these conditions on the coordinates of points A, B, C, and D, without attempting to solve the system yet.")
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, geometric conditions, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent geometric conditions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Enumerate all algebraic constraints on the coordinates of points A, B, C, and D imposed by the rhombus properties "
        "(equal side lengths, perpendicular diagonals bisected at the origin) and the hyperbola equation. Present these constraints explicitly and separately, "
        "avoiding any premature attempts to solve or simplify them.")
    cot_sc_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_3[i].id}, algebraic constraints, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent algebraic constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Parametric representation and relations (CoT)

    cot_instruction_1_1 = (
        "Sub-task 1: Derive a parametric representation of points on the hyperbola using hyperbolic functions, "
        "specifically expressing any point (x,y) on the hyperbola as x = sqrt(20)*cosh(t), y = sqrt(24)*sinh(t). "
        "Emphasize the domain and range of the parameter t and avoid introducing additional parameters at this stage.")
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_1], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, parametric representation, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Express the coordinates of points A and B in terms of parameters u and v using the hyperbolic parameterization from subtask_1, "
        "ensuring that points C and D are determined by the midpoint and symmetry conditions of the rhombus with diagonals intersecting at the origin. "
        "Explicitly incorporate the perpendicularity condition of the diagonals to relate u and v. Avoid treating u and v as independent without enforcing this relation.")
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0_2.content, thinking_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_2, thinking_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, parametric points A and B, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Validate the parametric forms of points A, B, C, and D by substituting them back into the hyperbola equation and the rhombus side length equality condition. "
        "Confirm that all constraints are satisfied symbolically, and explicitly state any relations or restrictions on parameters u and v that arise. "
        "Avoid skipping intermediate algebraic steps to ensure no loss of constraint information.")
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_0_3.content, thinking_1_1.content, thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_1_3[i]([taskInfo, thinking_0_3, thinking_1_1, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, validate parametric forms, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_3.append(answer)
        possible_thinkings_1_3.append(thinking)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and confirm parametric validation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 6: ", sub_tasks[-1])

    # Stage 2: Expressions for diagonals and side length equality (CoT and SC_CoT)

    cot_instruction_2_1 = (
        "Sub-task 1: Compute explicit expressions for the squared lengths of the diagonals AC and BD in terms of the parameters u and v derived previously. "
        "Present these expressions clearly and verify their consistency with the rhombus properties. Avoid approximations or heuristic simplifications at this stage.")
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2.content, thinking_1_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, thinking_1_3], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, diagonal lengths, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 7: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Derive the condition for the rhombus side length equality in terms of parameters u and v, relating the side length to the diagonal lengths. "
        "Explicitly incorporate the perpendicularity of the diagonals and the midpoint conditions. Present the resulting system of equations clearly, ensuring all constraints are simultaneously enforced.")
    cot_sc_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_2_2[i]([taskInfo, thinking_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_2[i].id}, side length equality, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_2.append(answer)
        possible_thinkings_2_2.append(thinking)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_thinkings_2_2, "Sub-task 2: Synthesize and confirm side length equality conditions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_2_3 = (
        "Sub-task 3: Express BD^2 explicitly as a function of a single parameter by substituting the relations between u and v from the perpendicularity and side length equality constraints. "
        "Identify the domain of this parameter consistent with all constraints and the hyperbola. Avoid treating parameters as independent or ignoring domain restrictions.")
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_2_1.content, thinking_2_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3([taskInfo, thinking_2_1, thinking_2_2], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_3.id}, BD^2 single parameter expression, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 9: ", sub_tasks[-1])

    # Stage 3: Rigorous constrained optimization and final analysis (Debate and Reflexion)

    debate_instruction_3_1 = (
        "Sub-task 1: Perform a rigorous constrained optimization to find the infimum (greatest lower bound) of BD^2 over the parameter domain identified. "
        "Use Lagrange multipliers or equivalent calculus-based methods to simultaneously enforce the hyperbola equation, the rhombus side length equality, and the perpendicularity of the diagonals. "
        "Provide detailed algebraic derivations and verify that all constraints remain satisfied throughout. Avoid heuristic guesses or treating variables as independent. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_3], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_3] + all_thinking_3_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, rigorous optimization, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_1[r].append(thinking)
            all_answer_3_1[r].append(answer)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final rigorous optimization answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 10: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 2: Analyze whether the infimum of BD^2 found in subtask_1 is attained (minimum) or only approached (supremum). "
        "Clarify the distinction between infimum and minimum in this context. Provide numeric or symbolic verification to support the conclusion, "
        "and explicitly state the greatest real number less than all possible values of BD^2 for the rhombi on the hyperbola. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_1] + all_thinking_3_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, infimum analysis, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_2[r].append(thinking)
            all_answer_3_2[r].append(answer)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final infimum analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 11: ", sub_tasks[-1])

    reflect_inst_3_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3_3 = (
        "Sub-task 3: Simplify and present the final exact expression or numerical approximation for the greatest real number less than all possible values of BD^2, "
        "ensuring the result is fully justified by the problem conditions and previous derivations. Include explicit checks for parameter feasibility and domain restrictions. "
        "Avoid skipping verification steps or leaving the result unsubstantiated. " + reflect_inst_3_3)
    cot_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_3 = self.max_round
    cot_inputs_3_3 = [taskInfo, thinking_3_1, thinking_3_2]
    subtask_desc_3_3 = {
        "subtask_id": "stage_3.subtask_3",
        "instruction": cot_reflect_instruction_3_3,
        "context": ["user query", thinking_3_1.content, thinking_3_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_3, answer_3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_reflect_instruction_3_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, final simplification, thinking: {thinking_3_3.content}; answer: {answer_3_3.content}")
    critic_inst_3_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_3_3):
        feedback, correct = await critic_agent_3_3([taskInfo, thinking_3_3], critic_inst_3_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_3.id}, round {i}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_3.extend([thinking_3_3, feedback])
        thinking_3_3, answer_3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_reflect_instruction_3_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, refining final simplification, thinking: {thinking_3_3.content}; answer: {answer_3_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3_3.content}; answer - {answer_3_3.content}")
    subtask_desc_3_3['response'] = {"thinking": thinking_3_3, "answer": answer_3_3}
    logs.append(subtask_desc_3_3)
    print("Step 12: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_3, answer_3_3, sub_tasks, agents)
    return final_answer, logs
