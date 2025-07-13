async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally define the problem using set theory and group actions. "
        "Represent the octagon vertices as the set {0,...,7} and colorings as subsets of vertices colored blue (with the rest red). "
        "Define the rotation group G as the cyclic group of order 8 acting by cyclic shifts on vertex indices. "
        "Precisely express the condition that there exists a non-identity rotation g in G such that the blue set B after rotation g is contained in the original red set R, i.e., B ⊆ R^g, or equivalently, B ∩ B^g = ∅. "
        "Clarify all notation and assumptions, including labeling vertices 0 to 7 in order and rotations acting as cyclic shifts. "
        "Avoid any conflation of this condition with invariance of B under rotation. This subtask sets the rigorous mathematical foundation for subsequent analysis."
    )

    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }

    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formalizing problem, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the condition B ∩ B^g = ∅ for some non-identity rotation g in G. "
        "For each rotation k in {1,...,7}, characterize the structure of the graph formed by edges between vertices i and i+k mod 8. "
        "Show that the condition B ∩ B^{r^k} = ∅ means B is an independent set in this graph. "
        "Identify the types of graphs formed for each k (e.g., 8-cycle, union of smaller cycles, or perfect matching). "
        "Emphasize that the problem reduces to counting subsets B that are independent sets in at least one of these graphs. "
        "Avoid any assumptions about invariance or fixed points. This subtask bridges the problem condition to combinatorial graph theory."
    )

    N_sc = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, analyzing rotation condition, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: For each rotation k in {1,...,7}, enumerate or compute the number of independent sets in the corresponding graph G_k on 8 vertices defined by edges between vertices i and i+k mod 8. "
        "Use known combinatorial formulas or direct enumeration to find |A_k|, where A_k is the set of subsets B with B ∩ B^{r^k} = ∅. "
        "Carefully document the structure of each G_k and the count of its independent sets. Avoid double counting or misinterpretation of the graphs. "
        "This subtask provides the building blocks for inclusion-exclusion."
    )

    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }

    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, counting independent sets per rotation graph, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Apply the principle of inclusion-exclusion to compute the size of the union |⋃_{k=1}^7 A_k|, i.e., the number of subsets B that are independent sets in at least one G_k. "
        "Carefully handle intersections of these sets by analyzing the intersection graphs or using combinatorial arguments. "
        "Avoid approximations or assumptions that could lead to undercounting or overcounting. "
        "This subtask yields the total count of valid blue sets B satisfying the problem condition."
    )

    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}

    for i in range(N_sc):
        thinking4, answer4 = await cot_sc_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, applying inclusion-exclusion, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4

    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[best_answer_4]
    answer4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Compute the probability that a random coloring satisfies the condition as |⋃_{k=1}^7 A_k| divided by the total number of subsets (2^8 = 256). "
        "Express this probability as a reduced fraction m/n with gcd(m,n) = 1. Calculate and return the final answer m + n. "
        "Verify the correctness of fraction reduction and arithmetic. Avoid errors in simplification or probability calculation. "
        "This subtask finalizes the solution."
    )

    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "CoT"
    }

    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, computing final probability and fraction, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_6 = (
        "Sub-task 6: Perform a verification and reflection step on the entire reasoning and counting process. "
        "Re-express the problem condition and confirm that the counting method correctly models the disjointness condition without conflating it with invariance. "
        "Cross-check intermediate counts and inclusion-exclusion steps for logical consistency. "
        "Optionally, test the approach on smaller analogous cases (e.g., smaller polygons) to validate correctness. "
        "This subtask ensures the solution's logical soundness and guards against previous errors."
    )

    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_6 = self.max_round

    all_thinking_6 = [[] for _ in range(N_rounds_6)]
    all_answer_6 = [[] for _ in range(N_rounds_6)]

    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Debate"
    }

    for r in range(N_rounds_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking_6[r-1] + all_answer_6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying reasoning, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking_6[r].append(thinking6)
            all_answer_6[r].append(answer6)

    final_thinking_6 = all_thinking_6[-1][0]
    final_answer_6 = all_answer_6[-1][0]
    sub_tasks.append(f"Sub-task 6 output: thinking - {final_thinking_6.content}; answer - {final_answer_6.content}")
    subtask_desc_6['response'] = {"thinking": final_thinking_6, "answer": final_answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    reflexion_instruction_7 = (
        "Sub-task 7: Synthesize the verified results and produce the final answer with confidence. "
        "Summarize the key logical steps, the counting approach, and the final probability. "
        "Confirm that the answer m + n is consistent with the problem requirements and previous verification. "
        "Provide a clear, concise conclusion. This subtask closes the workflow with a validated solution."
    )

    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": reflexion_instruction_7,
        "context": ["user query", final_thinking_6.content, final_answer_6.content],
        "agent_collaboration": "Reflexion"
    }

    thinking7, answer7 = await cot_agent_7([taskInfo, final_thinking_6, final_answer_6], reflexion_instruction_7, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, synthesizing final answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc_7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
