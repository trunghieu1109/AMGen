async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally define the problem parameters and constraints. "
        "Express the two-digit number n in base b as n = x*b + y with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "Translate the b-eautiful condition into the key equation (x + y)^2 = x*b + y. "
        "Explicitly note that n must be a perfect square and that √n = x + y is an integer. "
        "Establish the domain of n as [b, b^2 - 1]. Avoid assumptions beyond the problem statement and explicitly clarify all digit and base constraints, including that the leading digit x cannot be zero. "
        "This subtask sets the formal foundation for all subsequent analysis and enumeration."
    )
    subtask_id_1 = "subtask_1"
    print(f"Starting {subtask_id_1}")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": subtask_id_1,
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the equation (x + y)^2 = x*b + y to understand the relationship between digits x, y, and base b. "
        "Derive algebraic expressions or inequalities that can facilitate enumeration, such as expressing y in terms of x, b, and s = x + y, or bounding possible values of s. "
        "Identify digit bounds and perfect square conditions to prune the search space. Avoid premature enumeration; focus on algebraic insight and parameter relationships that will optimize the enumeration process. "
        "This subtask depends on the formal definitions and constraints established in subtask_1."
    )
    subtask_id_2 = "subtask_2"
    print(f"Starting {subtask_id_2}")
    N_sc = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc2 = {
        "subtask_id": subtask_id_2,
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    possible_thinkings_2 = []
    for i in range(N_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, analyzing algebraic relationships, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 2: Synthesize and choose the most consistent and correct algebraic insights to optimize enumeration."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3a = (
        "Sub-task 3a: Enumerate all bases b from 2 up to 30. For each base b, enumerate all digit pairs (x,y) with 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "For each pair, check if (x + y)^2 = x*b + y holds. Collect and output a precise count of b-eautiful integers for each base b in a clear tabular or list format (b, count). "
        "Embed or reference a computational routine (e.g., Python code or detailed pseudo-code) to perform this enumeration to guarantee correctness beyond LLM reasoning. "
        "Ensure no duplicates and strictly enforce digit constraints. This subtask depends on the algebraic insights from subtask_2 and the formal constraints from subtask_1."
    )
    subtask_id_3a = "subtask_3a"
    print(f"Starting {subtask_id_3a}")
    cot_sc_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.3) for _ in range(N_sc)]
    subtask_desc3a = {
        "subtask_id": subtask_id_3a,
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3a = []
    possible_thinkings_3a = []

    for i in range(N_sc):
        thinking3a, answer3a = await cot_sc_agents_3a[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3a[i].id}, enumerating b-eautiful integers, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a)
        possible_thinkings_3a.append(thinking3a)

    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3a = "Sub-task 3a: Synthesize and choose the most consistent and correct enumeration results for b-eautiful integers across bases 2 to 30."
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo] + possible_answers_3a + possible_thinkings_3a, final_instr_3a, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_instruction_3b = (
        "Sub-task 3b: Analyze the enumeration results from subtask_3a to identify the minimal base b for which the count of b-eautiful integers exceeds 10. "
        "Justify minimality by confirming that all smaller bases have counts at most 10. Provide explicit references to the enumeration data to support the conclusion. "
        "Avoid premature conclusions or assumptions without direct evidence from the enumeration table. This subtask depends explicitly on the enumeration output from subtask_3a."
    )
    subtask_id_3b = "subtask_3b"
    print(f"Starting {subtask_id_3b}")
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": subtask_id_3b,
        "instruction": cot_instruction_3b,
        "context": ["user query", thinking3a.content, answer3a.content],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, analyzing enumeration results, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Sub-task 4: Perform a rigorous verification of the enumeration data and minimal base identification. "
        "Cross-check each (b, count) entry line-by-line, confirming digit constraints and the b-eautiful condition for all counted pairs. "
        "Verify that no smaller base than the identified minimal base has more than ten b-eautiful integers. "
        "Use adversarial or multi-agent debate patterns to challenge and validate the enumeration and minimality claims, reducing the risk of hallucinations or errors. "
        "This subtask depends on the outputs of subtask_3a and subtask_3b."
    )
    subtask_id_4 = "subtask_4"
    print(f"Starting {subtask_id_4}")
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": subtask_id_4,
        "instruction": debate_instruction_4,
        "context": ["user query", thinking3a.content, answer3a.content, thinking3b.content, answer3b.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3a, answer3a, thinking3b, answer3b], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3a, answer3a, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying enumeration and minimality, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Synthesize debate results and confirm verification of enumeration and minimal base.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing verification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction_5 = (
        "Sub-task 5: Synthesize the initial findings from subtask_3b with the verification results from subtask_4 to produce a final, confident solution. "
        "State the minimal base b with more than ten b-eautiful integers, the exact count for that base, and confirm that all verification steps have been passed. "
        "Document any assumptions, limitations, or edge cases considered. This final subtask ensures the solution is fully grounded in explicit, verified data and reasoning."
    )
    subtask_id_5 = "subtask_5"
    print(f"Starting {subtask_id_5}")
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking3b, answer3b, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": subtask_id_5,
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking3b.content, answer3b.content, thinking4.content, answer4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, synthesizing final solution, thinking: {thinking5.content}; answer: {answer5.content}")
    critic_inst_5 = "Please review and provide limitations or confirm correctness. If correct, output exactly 'True' in 'correct'."
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], critic_inst_5, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining final solution, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
