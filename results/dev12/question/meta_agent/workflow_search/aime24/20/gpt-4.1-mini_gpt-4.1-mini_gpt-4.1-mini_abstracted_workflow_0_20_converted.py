async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # -------------------------------- Stage 1: Formal Definitions and Key Equation --------------------------------
    
    cot_instruction_1 = (
        "Sub-task 1: Formally define the representation of a two-digit number n in base b as n = x*b + y, "
        "with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. Express the condition that n is b-eautiful as x + y = √n, "
        "emphasizing that √n must be an integer and that n must be a perfect square. Avoid solving or enumerating solutions at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formal definition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Derive the key equation x*b + y = (x + y)^2 from the b-eautiful condition. "
        "Analyze integer and digit constraints on x, y, and b, explicitly stating domain restrictions: x ≥ 1, y ≥ 0, x,y < b, b ≥ 2, and s = x + y = √n. "
        "Clarify that n must be a perfect square and s is an integer digit sum. Avoid assumptions about solution counts or enumeration."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, derive key equation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent answer for the key equation and constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = (
        "Sub-task 3: Explicitly state and clarify all digit and base constraints, including ranges of x, y, and b, "
        "and implications on possible values of s = x + y. Emphasize leading digit x ≥ 1, y ≥ 0, digits < b, b ≥ 2, and n must be a perfect square. "
        "Avoid enumeration or counting in this step."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1.content, answer1.content], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, clarify digit/base constraints, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    # -------------------------------- Stage 2: Enumeration and Counting --------------------------------
    
    cot_reflect_instruction_4 = (
        "Sub-task 4: For a fixed base b, explicitly enumerate all possible digit pairs (x,y) with 1 ≤ x ≤ b-1, 0 ≤ y ≤ b-1. "
        "Check if x*b + y = (x + y)^2 holds. Record all valid b-eautiful numbers n = x*b + y with digit pairs and sums. "
        "Implement as clear step-by-step procedure or pseudocode, ensuring no solutions missed. Store results in structured format."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2.content, answer2.content, thinking3.content, answer3.content], cot_reflect_instruction_4, is_sub_task=True)
    agents.append(f"CoT-Reflexion agent {cot_agent_4.id}, enumerate digit pairs for base b, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_reflect_instruction_5 = (
        "Sub-task 5: Develop an algebraic or combinatorial counting method to determine the number of b-eautiful numbers for given base b without exhaustive enumeration. "
        "Include explicit digit bound checks after divisibility or modular conditions. Design method to facilitate comparison with enumeration results."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2.content, answer2.content, thinking3.content, answer3.content], cot_reflect_instruction_5, is_sub_task=True)
    agents.append(f"CoT-Reflexion agent {cot_agent_5.id}, develop counting method, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    debate_instruction_6 = (
        "Sub-task 6: Validate the counting method by applying it to multiple example bases including b=13. "
        "Compare counts from counting method with explicit enumerations from Sub-task 4. Identify and resolve discrepancies. "
        "Document validation process and results in detail. Avoid accepting counting method without numeric verification. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", thinking4.content, answer4.content, thinking5.content, answer5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4.content, answer4.content, thinking5.content, answer5.content], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4.content, answer4.content, thinking5.content, answer5.content] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validate counting method, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Cross-validate counting method with enumerations and finalize validation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    # -------------------------------- Stage 3: Iteration, Exhaustive Enumeration, Cross-validation, Finalization --------------------------------
    
    cot_reflect_instruction_7 = (
        "Sub-task 7: Iterate over increasing bases b ≥ 2, applying the validated counting method to determine the number of b-eautiful numbers for each base. "
        "Maintain a running log recording each base alongside its count and enumerated b-eautiful numbers where feasible. "
        "Identify candidate bases where count exceeds ten. Avoid skipping bases or relying on symbolic reasoning alone. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking5.content, answer5.content, thinking6.content, answer6.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5.content, answer5.content, thinking6.content, answer6.content], cot_reflect_instruction_7, is_sub_task=True)
    agents.append(f"CoT-Reflexion agent {cot_agent_7.id}, iterate bases and count b-eautiful numbers, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_reflect_instruction_8 = (
        "Sub-task 8: For candidate bases near threshold (e.g., bases 15 to 21), perform exhaustive enumeration as in Sub-task 4. "
        "Explicitly list all valid (x,y) pairs and corresponding n values. Confirm count > 10 and no smaller base meets criterion. "
        "Document enumerations and verification results in detail. Avoid accepting minimal base claims without explicit numeric confirmation."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", thinking7.content, answer7.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7.content, answer7.content], cot_reflect_instruction_8, is_sub_task=True)
    agents.append(f"CoT-Reflexion agent {cot_agent_8.id}, exhaustive enumeration for candidate bases, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    debate_instruction_9 = (
        "Sub-task 9: Cross-validate counts and enumerations from Sub-tasks 7 and 8 by comparing algebraic counting method results with explicit enumerations for critical bases. "
        "Resolve discrepancies through detailed analysis and re-execution if necessary. Ensure consensus on minimal base determination. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", thinking7.content, answer7.content, thinking8.content, answer8.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking7.content, answer7.content, thinking8.content, answer8.content], debate_instruction_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking7.content, answer7.content, thinking8.content, answer8.content] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validate counts and enumerations, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Final cross-validation and consensus on minimal base.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    cot_reflect_instruction_10 = (
        "Sub-task 10: Finalize the least integer base b ≥ 2 for which there are more than ten b-eautiful numbers. "
        "Present final answer supported by explicit enumerations, validated counts, and cross-checked results. "
        "Include concrete examples such as the 11th b-eautiful number for the minimal base to demonstrate correctness. "
        "Avoid unsupported claims or acceptance without concrete numeric evidence."
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_reflect_instruction_10,
        "context": ["user query", thinking9.content, answer9.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking9.content, answer9.content], cot_reflect_instruction_10, is_sub_task=True)
    agents.append(f"CoT-Reflexion agent {cot_agent_10.id}, finalize minimal base and examples, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
