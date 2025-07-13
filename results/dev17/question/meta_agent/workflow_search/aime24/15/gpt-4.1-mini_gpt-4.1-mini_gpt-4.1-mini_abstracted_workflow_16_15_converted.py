async def forward_15(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Problem Domain and Data Enumeration (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state the domain of the problem: the set of 900 residents of Aimeville and the four items they own (diamond ring, golf clubs, garden spade, candy hearts). "
        "Emphasize that candy hearts are owned by all residents, making it a universal set in this context. Avoid making any assumptions about the distribution of ownership beyond this fact."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, domain identification, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Enumerate the sizes of the individual sets: 195 own diamond rings, 367 own golf clubs, 562 own garden spades, and all 900 own candy hearts. "
        "Also, state the counts of residents owning exactly two items (437) and exactly three items (234). Explicitly clarify that these counts include candy hearts as one of the items, and avoid assuming these counts include residents owning more than the specified number of items."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, data enumeration, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Clarify and confirm the interpretation of 'exactly k items owned' for k=1,2,3,4, ensuring these counts represent disjoint subsets of residents owning exactly that number of items, including candy hearts. "
        "Explicitly introduce the variable for the number of residents owning exactly one item (candy hearts only), and avoid assuming it to be zero. This subtask should also clarify that the problem reduces to analyzing intersections among the three non-candy sets with candy hearts as universal."
    )
    N_sc = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, clarify exactly k items, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent interpretation for exactly k items owned including candy hearts.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Subtask 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Variable Definition and Equation Setup
    cot_instruction_1_1 = (
        "Sub-task 1: Formally define variables representing the sizes of all disjoint subsets of residents owning exactly one, exactly two, exactly three, and exactly four items. "
        "These subsets should be defined in terms of the four sets (D, G, S, C), with candy hearts (C) as universal. Avoid overly granular partitioning; instead, collapse into the four main partitions E1, E2, E3, and E4 representing exactly one, two, three, and four items owned respectively. "
        "Explicitly include the variable for residents owning only candy hearts (exactly one item)."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, define variables E1-E4, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Express the given constraints (total sizes of sets D, G, S, and C, counts of exactly two and exactly three items) in terms of the defined disjoint subset variables E1, E2, E3, E4. "
        "Set up a system of equations using inclusion-exclusion principles and the definitions of exactly k items owned, ensuring that counts of exactly two and exactly three items exclude residents owning more items. Avoid oversimplification or conflation of these counts with residents owning more than k items."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, set up equations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent system of equations for E1-E4.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Analyze the system of equations to isolate and simplify expressions involving the quadruple intersection size (E4). "
        "Perform symbolic manipulation to express E4 in terms of known quantities and other variables. Include explicit non-negativity constraints for all variables and check for internal consistency. "
        "If contradictions or negative values arise, flag these for re-examination of assumptions or data."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, analyze system, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and simplify expressions for E4 with constraints.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2: Solve and Verify
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Solve the simplified system of equations to determine the exact number of residents who own all four items (the quadruple intersection E4). "
        "Perform step-by-step solving with clear variable substitutions and arithmetic. Avoid accepting solutions without verifying that all variables are non-negative and consistent with the given set sizes and counts."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_3], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, solve system, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 1: Synthesize and finalize solution for E4.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    debate_instruction_2_2 = (
        "Sub-task 2: Verify the solution obtained for E4 against all given constraints, including total population, individual set sizes, and counts of exactly one, two, and three items owned. "
        "Perform logical feasibility checks and ensure no contradictions or negative values exist. If inconsistencies are found, prompt a re-examination of earlier assumptions or the system of equations. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2_2 = self.max_round
    all_thinking_2_2 = [[] for _ in range(N_max_2_2)]
    all_answer_2_2 = [[] for _ in range(N_max_2_2)]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": debate_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_2):
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_1], debate_instruction_2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1] + all_thinking_2_2[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verify solution, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_2[r].append(thinking_i)
            all_answer_2_2[r].append(answer_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1], "Sub-task 2: Final verification and consistency check of solution.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_instruction_2_3 = (
        "Sub-task 3: Interpret the verified solution logically and provide the final answer: the number of residents owning all four items. "
        "Summarize the reasoning process and highlight how the solution respects all constraints and assumptions clarified earlier."
    )
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3([taskInfo, thinking_2_2], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_3.id}, interpret final solution, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Stage 2 Subtask 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
