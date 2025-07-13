async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: select_and_verify_elements_under_constraints

    # Sub-task 1: Identify and enumerate all possible digit assignments that could satisfy sum constraints
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and enumerate all possible digit assignments to the 2x3 grid cells (digits 0-9), "
        "considering leading zeros allowed and no digit repetition restrictions, to understand the problem domain and constraints. "
        "Avoid premature exclusions."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, enumerating digit assignments, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Verify problem constraints explicitly, confirm sum conditions and assumptions
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Verify the problem constraints explicitly: confirm that the sum of the two 3-digit numbers formed by the rows equals 999, "
        "and the sum of the three 2-digit numbers formed by the columns equals 99. Check the example grid and clarify assumptions about leading zeros, digit repetition, and interpretation."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, verifying constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent verification of problem constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: derive_and_validate_representations

    # Sub-task 1: Derive formal algebraic representations of sum constraints
    cot_instruction_1_1 = (
        "Sub-task 1: Derive formal algebraic equations for the problem using variables a,b,c,d,e,f for the grid digits. "
        "Express row sums as 100a+10b+c and 100d+10e+f summing to 999, and column sums as 10a+d, 10b+e, 10c+f summing to 99. "
        "Validate correctness and consistency of these equations."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, deriving algebraic equations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Sub-task 2: Validate derived equations by substituting example grid digits
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Validate the algebraic equations by substituting the example grid digits (0,0,8,9,9,1) to confirm sums equal 999 and 99. "
        "Ensure correctness before solving."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, validating equations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent validation of algebraic equations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 2: infer_compute_parameters_from_composite_data

    # Sub-task 1: Analyze system of equations to infer digit relationships and constraints
    cot_instruction_2_1 = (
        "Sub-task 1: Analyze the system of equations derived to infer relationships between digits a,b,c,d,e,f. "
        "Simplify equations, express some digits in terms of others, and identify digit value constraints (0-9). "
        "Goal is to reduce search space for enumeration."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, analyzing equations, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Sub-task 2: Compute all possible digit assignments satisfying simplified system and constraints
    cot_sc_instruction_2_2 = (
        "Sub-task 2: Enumerate and compute all digit assignments (a,b,c,d,e,f) satisfying the simplified system and digit constraints (0-9). "
        "Prune invalid assignments early and ensure sum conditions hold. Output all valid 2x3 digit grids."
    )
    N_sc_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_2):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, enumerating valid digit assignments, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and choose the most consistent set of valid digit assignments.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Stage 3: decompose_simplify_and_sum_components

    # Sub-task 1: Verify each valid digit assignment explicitly and count total valid configurations
    debate_instruction_3_1 = (
        "Sub-task 1: Decompose each candidate digit assignment into row and column numbers, verify sums explicitly, "
        "and count total number of distinct valid configurations satisfying both sum conditions exactly. "
        "Ensure no invalid solutions are counted."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_rounds_3_1)]
    all_answer_3_1 = [[] for _ in range(N_rounds_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying and counting valid configs, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3_1[r].append(thinking_i)
            all_answer_3_1[r].append(answer_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 1: Final verification and counting of valid digit assignments.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    # Sub-task 2: Reflect on solution process and results to confirm correctness and completeness
    reflect_instruction_3_2 = (
        "Sub-task 2: Reflect on the entire solution process and results. Check for overlooked edge cases, verify assumptions about leading zeros and digit repetition, "
        "and ensure the final count aligns with problem constraints. Provide a summary of findings and insights."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking_3_1, answer_3_1]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": reflect_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflect_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, reflecting on solution, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    for i in range(N_max_3_2):
        feedback, correct = await critic_agent_3_2([taskInfo, thinking_3_2, answer_3_2], "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, answer_3_2, feedback])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflect_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining reflection, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
