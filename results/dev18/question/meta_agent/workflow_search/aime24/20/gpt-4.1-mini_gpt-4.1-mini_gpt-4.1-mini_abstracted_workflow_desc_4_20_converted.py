async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 0_1: Derive the formal equation relating digits x,y and base b for a two-digit number n in base b, "
        "expressing the b-eautiful condition as (x + y)^2 = x*b + y, with digit constraints 1 <= x <= b-1 and 0 <= y <= b-1. "
        "Clearly state all assumptions and ensure the equation is suitable for further algebraic and computational analysis."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving formal equation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 0_2: Validate the digit constraints and domain of n, confirming n lies between b and b^2 - 1 to ensure exactly two digits in base b, "
        "and that the sum of digits equals the integer square root of n. Explicitly state these domain restrictions and their implications."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, validating digit constraints and domain, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1_1: For each base b from 2 up to 20, enumerate all possible digit pairs (x,y) satisfying 1 <= x <= b-1 and 0 <= y <= b-1, "
        "and identify those pairs that satisfy (x + y)^2 = x*b + y. Produce a detailed table listing each valid (x,y), corresponding n = x*b + y, and sum s = x + y. "
        "This enumeration must be explicit and computationally verified."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating valid digit pairs for bases 2 to 20, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1_1: Synthesize and choose the most consistent enumeration of valid digit pairs and counts per base.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    debate_instruction_1_2 = (
        "Sub-task 1_2: Verify for each identified pair (x,y) from enumeration that n = x*b + y is indeed two-digit in base b and sum of digits equals sqrt(n), "
        "ensuring no extraneous or invalid solutions are included. Report discrepancies or edge cases found during verification. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking_1_1], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_1] + all_thinking_1_2[r-1]
                thinking_d, answer_d = await agent(input_infos, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying enumeration correctness, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_1_2[r].append(thinking_d)
            all_answer_1_2[r].append(answer_d)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + all_thinking_1_2[-1], "Sub-task 1_2: Final verification and correction of enumeration results.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    debate_instruction_2_1 = (
        "Sub-task 2_1: Algebraically decompose (x + y)^2 = x*b + y to express y in terms of x and b or vice versa, "
        "to simplify the search for solutions. Identify patterns, bounds, or constraints on x, y, and b to reduce computational complexity. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking_0_1], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_1] + all_thinking_2_1[r-1]
                thinking_d, answer_d = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, algebraic decomposition, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_2_1[r].append(thinking_d)
            all_answer_2_1[r].append(answer_d)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 2_1: Final synthesis of algebraic decomposition and constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    debate_instruction_2_2 = (
        "Sub-task 2_2: Analyze digit constraints and algebraic expressions from 2_1 to identify patterns or bounds on x, y, and b that facilitate efficient enumeration. "
        "Use these insights to optimize enumeration and reduce brute force checks. Document optimizations clearly. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2_2 = self.max_round
    all_thinking_2_2 = [[] for _ in range(N_max_2_2)]
    all_answer_2_2 = [[] for _ in range(N_max_2_2)]
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": debate_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_2):
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking_2_1], debate_instruction_2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1] + all_thinking_2_2[r-1]
                thinking_d, answer_d = await agent(input_infos, debate_instruction_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing constraints and optimizations, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_2_2[r].append(thinking_d)
            all_answer_2_2[r].append(answer_d)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1], "Sub-task 2_2: Final synthesis of enumeration optimizations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2_2: ", sub_tasks[-1])

    cot_sc_instruction_3_1a = (
        "Sub-task 3_1a: Implement enumeration logic programmatically, incorporating algebraic simplifications and optimizations from stage 2, "
        "to enumerate all valid b-eautiful numbers for each base b from 2 up to 20. Output a comprehensive table of valid (x,y), n, and counts per base. "
        "Include partial numeric examples for bases 10 and 15 to validate the method."
    )
    N_sc_3_1a = self.max_sc
    cot_agents_3_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_3_1a)]
    possible_answers_3_1a = []
    possible_thinkings_3_1a = []
    subtask_desc_3_1a = {
        "subtask_id": "subtask_3_1a",
        "instruction": cot_sc_instruction_3_1a,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3_1a):
        thinking_i, answer_i = await cot_agents_3_1a[i]([taskInfo, thinking_2_2], cot_sc_instruction_3_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1a[i].id}, enumerating b-eautiful numbers with optimizations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1a.append(answer_i)
        possible_thinkings_3_1a.append(thinking_i)
    final_decision_agent_3_1a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1a, answer_3_1a = await final_decision_agent_3_1a([taskInfo] + possible_thinkings_3_1a, "Sub-task 3_1a: Synthesize enumeration results and counts per base.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3_1a output: thinking - {thinking_3_1a.content}; answer - {answer_3_1a.content}")
    subtask_desc_3_1a['response'] = {"thinking": thinking_3_1a, "answer": answer_3_1a}
    logs.append(subtask_desc_3_1a)
    print("Step 3_1a: ", sub_tasks[-1])

    cot_reflect_instruction_3_1b = (
        "Sub-task 3_1b: Validate completeness and correctness of enumeration results from 3_1a by cross-checking counts, "
        "ensuring no valid solutions are missed and no extraneous solutions included. Perform consistency checks and document corrections or confirmations. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1b = self.max_round
    cot_inputs_3_1b = [taskInfo, thinking_3_1a, answer_3_1a]
    subtask_desc_3_1b = {
        "subtask_id": "subtask_3_1b",
        "instruction": cot_reflect_instruction_3_1b,
        "context": ["user query", thinking_3_1a.content, answer_3_1a.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_1b, answer_3_1b = await cot_agent_3_1b(cot_inputs_3_1b, cot_reflect_instruction_3_1b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1b.id}, validating enumeration results, thinking: {thinking_3_1b.content}; answer: {answer_3_1b.content}")
    for i in range(N_max_3_1b):
        feedback, correct = await critic_agent_3_1b([taskInfo, thinking_3_1b], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_1b.extend([thinking_3_1b, feedback])
        thinking_3_1b, answer_3_1b = await cot_agent_3_1b(cot_inputs_3_1b, cot_reflect_instruction_3_1b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1b.id}, refining validation, thinking: {thinking_3_1b.content}; answer: {answer_3_1b.content}")
    sub_tasks.append(f"Sub-task 3_1b output: thinking - {thinking_3_1b.content}; answer - {answer_3_1b.content}")
    subtask_desc_3_1b['response'] = {"thinking": thinking_3_1b, "answer": answer_3_1b}
    logs.append(subtask_desc_3_1b)
    print("Step 3_1b: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 3_2: Analyze validated enumeration data to identify the smallest base b >= 2 for which the count of b-eautiful numbers exceeds ten. "
        "Include checks for bases immediately below and above the candidate to avoid off-by-one errors. Provide a clear, data-supported conclusion with justification. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "subtask_3_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1b.content, answer_3_1b.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking_3_1b, answer_3_1b], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_1b, answer_3_1b] + all_thinking_3_2[r-1]
                thinking_d, answer_d = await agent(input_infos, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing minimal base, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_3_2[r].append(thinking_d)
            all_answer_3_2[r].append(answer_d)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1], "Sub-task 3_2: Final decision on minimal base with more than ten b-eautiful numbers.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3_2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3_2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
