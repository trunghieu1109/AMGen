async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive and validate the formal mathematical representation of the problem. "
        "Express the two-digit number n in base b as n = x*b + y with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "Formulate the key equation (x + y)^2 = x*b + y that characterizes b-eautiful integers. "
        "Explicitly confirm that sqrt(n) is an integer and that n has exactly two digits in base b (i.e., n < b^2). "
        "Avoid assuming any properties not given, such as digit ordering beyond standard base representation. "
        "Clearly state all assumptions and constraints to prevent ambiguity.")
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving and validating representations, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    reflexion_instruction_1 = (
        "Sub-task 2: Based on the derived equation (x + y)^2 = x*b + y, manipulate it to express y in terms of x, b, and s = x + y, or vice versa. "
        "Develop a method to enumerate all valid (x,y) digit pairs for a given base b that satisfy the equation and digit constraints. "
        "Strictly enforce digit bounds (1 ≤ x ≤ b-1, 0 ≤ y ≤ b-1) and integrality conditions. "
        "Avoid brute force at this stage but prepare the formula-based approach for enumeration. "
        "Provide clear instructions on handling integer divisibility and inequalities to avoid invalid digit values.")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": reflexion_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo, thinking_0, answer_0], reflexion_instruction_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, deriving composite measure, thinking: {thinking_1.content}; answer: {answer_1.content}")

    max_reflect_rounds = self.max_round
    cot_inputs_1 = [taskInfo, thinking_0, answer_0, thinking_1, answer_1]
    for i in range(max_reflect_rounds):
        feedback_1, correct_1 = await critic_agent_1(cot_inputs_1, "Please review and provide limitations or errors in the solution above. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback_1.content}; correctness: {correct_1.content}")
        if correct_1.content == "True":
            break
        cot_inputs_1.extend([thinking_1, answer_1, feedback_1])
        thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining composite measure, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_3a = (
        "Sub-task 3a: Perform a detailed worked example for base b=13. "
        "For each possible sum s = x + y, enumerate all candidate digit pairs (x,y) that satisfy (x + y)^2 = x*13 + y and digit constraints. "
        "Explicitly list all valid pairs, verify digit bounds, and confirm that each corresponds to a distinct two-digit integer n = x*13 + y. "
        "Document the enumeration process step-by-step to serve as a verification reference and debugging tool.")
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_3a, answer_3a = await cot_agent_3a([taskInfo, thinking_1, answer_1], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, detailed enumeration for b=13, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 2a: ", sub_tasks[-1])

    cot_instruction_3b = (
        "Sub-task 3b: Generate an explicit enumeration table of counts of b-eautiful integers for bases b = 2 through b = 20. "
        "For each base, count the number of valid (x,y) pairs satisfying the equation and digit constraints using the formula-based method validated in subtask_3a. "
        "Cross-check with brute-force enumeration over all digit pairs to ensure accuracy. "
        "Output the full table with counts and highlight bases where the count exceeds ten. "
        "Ensure no double counting or invalid pairs are included.")
    N_sc_3b = self.max_sc
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3b)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_3a.content, answer_3a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3b):
        thinking_3b, answer_3b = await cot_agents_3b[i]([taskInfo, thinking_1, answer_1, thinking_3a, answer_3a], cot_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, enumerating counts for b=2..20, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
        possible_answers_3b.append(answer_3b.content)
        thinkingmapping_3b[answer_3b.content] = thinking_3b
        answermapping_3b[answer_3b.content] = answer_3b
    best_answer_3b = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking_3b = thinkingmapping_3b[best_answer_3b]
    answer_3b = answermapping_3b[best_answer_3b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 2b: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Select the minimal base b from the enumeration table where the count of b-eautiful integers exceeds ten. "
        "Verify that for all smaller bases, the count is ten or fewer. "
        "Cross-check the counts using both formula-based and brute-force methods. "
        "Pass the detailed enumeration data and worked example as context to this verification step. "
        "Avoid premature conclusions by thoroughly validating all candidate bases and their counts.")
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking_3b.content, answer_3b.content, thinking_3a.content, answer_3a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking_4, answer_4 = await cot_agents_4[i]([taskInfo, thinking_3b, answer_3b, thinking_3a, answer_3a], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, verifying minimal base, thinking: {thinking_4.content}; answer: {answer_4.content}")
        possible_answers_4.append(answer_4.content)
        thinkingmapping_4[answer_4.content] = thinking_4
        answermapping_4[answer_4.content] = answer_4
    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking_4 = thinkingmapping_4[best_answer_4]
    answer_4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Conduct a reflexion and debate stage where multiple agents independently verify the enumeration results and minimal base selection. "
        "Reconcile any discrepancies found in counts or assumptions. "
        "Confirm the correctness of the final minimal base b and the count of b-eautiful integers at that base. "
        "Provide a final answer with detailed justification and verification logs. "
        "This step ensures robustness and consensus before concluding.")
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking_4.content, answer_4.content, thinking_3b.content, answer_3b.content, thinking_3a.content, answer_3a.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4, thinking_3b, answer_3b, thinking_3a, answer_3a], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4, thinking_3b, answer_3b, thinking_3a, answer_3a] + all_thinking5[r-1] + all_answer5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking5[r].append(thinking_5)
            all_answer5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Final decision on minimal base b and count of b-eautiful integers.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
