async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Formally define a two-digit number n in base b as n = x*b + y, with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "Express the b-eautiful condition as s = x + y = sqrt(n), leading to s² = x*b + y. "
        "Clarify all assumptions about digit ranges and base, ensuring no ambiguity. Avoid heuristic assumptions; focus on precise algebraic formulation and validation of digit constraints.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving formal representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Validate the example that 81 is 13-eautiful by converting 81 to base 13, confirming digits (6 and 3), their sum (9), and that 9² = 81. "
        "This checks correctness of the derived formula and assumptions from subtask_1.")
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, validating example, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: For a fixed base b, enumerate all digit pairs (x,y) with 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "For each pair, compute s = x + y and n = x*b + y, then check if s² = n. Collect all such pairs satisfying the b-eautiful condition. "
        "Ensure exhaustive and exact enumeration without heuristic shortcuts, carefully handling digit bounds and avoiding double counting.")
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating b-eautiful numbers, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent enumeration of b-eautiful numbers for fixed base b.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Develop and implement a programmatic method to enumerate and count b-eautiful numbers for all bases b from 2 to 40. "
        "Iterate sums s from 1 to 2(b-1), use s² = x*b + y with y = s - x, solve for integer x,y within digit bounds. "
        "Tabulate counts of valid (x,y) pairs for each base. Produce a reliable dataset to support rigorous verification.")
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
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, enumerating counts for bases 2 to 40, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent counting method and enumeration results for bases 2 to 40.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_2_1 = (
        "Sub-task 1: Analyze the equation s² = x*b + y with s = x + y and digit constraints. "
        "Rewrite using y = s - x to isolate variables and derive necessary and sufficient conditions for integer solutions (x,y) within digit bounds. "
        "This aids understanding solution structure and guides enumeration. Avoid speculative assumptions; base conclusions on algebraic derivations.")
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_rounds_2_1)]
    all_answer_2_1 = [[] for _ in range(N_rounds_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing equation, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Synthesize debate results on equation analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflexion_instruction_2_2 = (
        "Sub-task 2: Using algebraic insights and enumeration results, analyze how the count of b-eautiful numbers changes with base b. "
        "Identify patterns or monotonicity without assuming linear growth. Cross-reference enumeration data from stage_1.subtask_2 to validate or refute conjectures. "
        "Prepare for precise identification of minimal base with more than ten b-eautiful numbers.")
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1, thinking_1_2, answer_1_2]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflexion_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflexion_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, analyzing counts and patterns, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide limitations of the analysis. If correct, output exactly 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflexion_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining analysis, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate enumeration data from stage_1.subtask_2 for bases b = 2 to 40. "
        "For each base, count total b-eautiful numbers. Identify smallest base b with count > 10. "
        "Present counts in clear tabular or structured format to support verification.")
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_2, answer_2_2, thinking_1_2, answer_1_2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating counts, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_instruction_3_2 = (
        "Sub-task 2: Verify correctness of identified minimal base b by rechecking counts and b-eautiful numbers for that base and bases immediately below and above. "
        "Confirm count > 10 for identified base and ≤ 10 for smaller bases. Provide explicit examples of b-eautiful numbers to support conclusion. "
        "Ensure rigorous, data-backed verification addressing edge cases and off-by-one errors.")
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2([taskInfo, thinking_3_1, answer_3_1], cot_instruction_3_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_2.id}, verifying minimal base, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    cot_instruction_3_3 = (
        "Sub-task 3: Synthesize results to provide final answer: least integer base b ≥ 2 with more than ten b-eautiful integers. "
        "Include concise explanation of reasoning, enumeration method, verification steps, and example b-eautiful numbers. "
        "Address subtleties or edge cases encountered. Integrate all prior findings into a coherent, rigorously justified conclusion.")
    cot_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_3 = {
        "subtask_id": "stage_3.subtask_3",
        "instruction": cot_instruction_3_3,
        "context": ["user query", thinking_3_2.content, answer_3_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_3, answer_3_3 = await cot_agent_3_3([taskInfo, thinking_3_2, answer_3_2], cot_instruction_3_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_3.id}, synthesizing final answer, thinking: {thinking_3_3.content}; answer: {answer_3_3.content}")
    sub_tasks.append(f"Sub-task 3.3 output: thinking - {thinking_3_3.content}; answer - {answer_3_3.content}")
    subtask_desc_3_3['response'] = {"thinking": thinking_3_3, "answer": answer_3_3}
    logs.append(subtask_desc_3_3)
    print("Step 3.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_3, answer_3_3, sub_tasks, agents)
    return final_answer, logs
