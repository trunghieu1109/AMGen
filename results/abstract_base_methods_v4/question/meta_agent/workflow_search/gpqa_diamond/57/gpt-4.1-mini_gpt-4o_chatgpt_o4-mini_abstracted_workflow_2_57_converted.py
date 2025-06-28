async def forward_57(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent_base = LLMAgentBase
    N_sc = self.max_sc
    N_round = self.max_round

    cot_temperature_low = 0.0
    cot_temperature_mid = 0.5

    # Stage 1: Define and Analyze Regularization and Theories

    # Sub-task 1a: Define regularization in quantum physical theories
    cot_instruction_1a = (
        "Sub-task 1a: Define the concept of regularization in quantum physical theories, "
        "explicitly including ultraviolet (UV) divergences and the need for regularization at high energies."
    )
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1a = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_low)
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, defining quantum regularization, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    # Sub-task 1b: Define regularization in classical physical theories
    cot_instruction_1b = (
        "Sub-task 1b: Define the concept of regularization in classical physical theories, "
        "explicitly including classical divergences such as the infinite self-energy of point charges in Classical Electrodynamics."
    )
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1b = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_low)
    thinking1b, answer1b = await cot_agent_1b([taskInfo], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, defining classical regularization, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    # Sub-task 2: Analyze Superstring Theory (depends on 1a,1b) with SC-CoT
    cot_instruction_2 = (
        "Sub-task 2: Analyze Superstring Theory’s fundamental nature and high-energy behavior, "
        "focusing on whether it requires regularization in the quantum context, considering its extended object nature, "
        "based on definitions from Sub-tasks 1a and 1b."
    )
    cot_agents_2 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_mid) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1a, answer1a, thinking1b, answer1b],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing Superstring Theory, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze Quantum Electrodynamics (QED) (depends on 1a,1b) with SC-CoT
    cot_instruction_3 = (
        "Sub-task 3: Analyze Quantum Electrodynamics (QED) regarding its high-energy behavior and the necessity of regularization due to quantum UV divergences, "
        "based on definitions from Sub-tasks 1a and 1b."
    )
    cot_agents_3 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_mid) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1a, answer1a, thinking1b, answer1b],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing QED, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze Quantum Chromodynamics (QCD) (depends on 1a,1b) with SC-CoT
    cot_instruction_4 = (
        "Sub-task 4: Analyze Quantum Chromodynamics (QCD) regarding its high-energy behavior and the necessity of regularization due to quantum UV divergences, "
        "based on definitions from Sub-tasks 1a and 1b."
    )
    cot_agents_4 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_mid) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking1a, answer1a, thinking1b, answer1b],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzing QCD, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Analyze Classical Electrodynamics with Debate (depends on 1a,1b)
    debate_instruction_5 = (
        "Sub-task 5: Analyze Classical Electrodynamics’ behavior at high energies, including a structured debate between two agents: "
        "one arguing no need for regularization due to classical theory status, and another highlighting classical divergences such as infinite self-energy, followed by reconciliation, "
        "based on definitions from Sub-tasks 1a and 1b."
    )
    debate_agents_5 = [cot_agent_base(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=cot_temperature_mid) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(N_round)]
    all_answer5 = [[] for _ in range(N_round)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking1a, answer1a, thinking1b, answer1b],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking1a, answer1a, thinking1b, answer1b], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking1a, answer1a, thinking1b, answer1b] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating Classical Electrodynamics regularization, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=cot_temperature_low)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on Classical Electrodynamics regularization after debate.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding Classical Electrodynamics regularization, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Integration, Identification, Clarification, and Final Selection

    # Sub-task 6: Integrate and review all analyses (2-5) with Reflexion
    cot_reflect_instruction_6 = (
        "Sub-task 6: Integrate and review all analyses from Sub-tasks 2 to 5, explicitly considering both quantum and classical divergences, "
        "to ensure no edge cases or ambiguities remain regarding the need for regularization."
    )
    cot_agent_6 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_low)
    critic_agent_6 = cot_agent_base(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=cot_temperature_low)
    cot_inputs_6 = [taskInfo, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, integrating analyses, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_round):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6],
                                                  "Please review the integration of analyses and identify any missing considerations or ambiguities.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining integration, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc_6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Identify all theories that never require regularization
    cot_instruction_7 = (
        "Sub-task 7: Identify all physical theories from the given choices that never require regularization at high energies, "
        "clearly distinguishing between classical and quantum contexts and noting multiple valid answers if applicable, "
        "based on the integrated analysis from Sub-task 6."
    )
    cot_agent_7 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_low)
    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6, answer6],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, identifying candidate theories, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc_7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8a: Clarify question scope and handling of multiple valid answers
    cot_instruction_8a = (
        "Sub-task 8a: Clarify the scope of the question (classical, quantum, or both) and determine how to handle multiple valid answers "
        "based on the identified candidate theories from Sub-task 7."
    )
    cot_agent_8a = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_low)
    subtask_desc_8a = {
        "subtask_id": "subtask_8a",
        "instruction": cot_instruction_8a,
        "context": ["user query", thinking7, answer7],
        "agent_collaboration": "CoT"
    }
    thinking8a, answer8a = await cot_agent_8a([taskInfo, thinking7, answer7], cot_instruction_8a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8a.id}, clarifying question scope, thinking: {thinking8a.content}; answer: {answer8a.content}")
    sub_tasks.append(f"Sub-task 8a output: thinking - {thinking8a.content}; answer - {answer8a.content}")
    subtask_desc_8a['response'] = {"thinking": thinking8a, "answer": answer8a}
    logs.append(subtask_desc_8a)
    print("Step 8a: ", sub_tasks[-1])

    # Sub-task 8b: Select best answer(s) and map to multiple-choice option(s)
    cot_instruction_8b = (
        "Sub-task 8b: Select the best answer(s) to the multiple-choice question based on the clarified scope from Sub-task 8a "
        "and identified candidate theories from Sub-task 7, mapping them to the corresponding letter choice(s) (A, B, C, or D)."
    )
    cot_agent_8b = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_temperature_low)
    subtask_desc_8b = {
        "subtask_id": "subtask_8b",
        "instruction": cot_instruction_8b,
        "context": ["user query", thinking7, answer7, thinking8a, answer8a],
        "agent_collaboration": "CoT"
    }
    thinking8b, answer8b = await cot_agent_8b([taskInfo, thinking7, answer7, thinking8a, answer8a], cot_instruction_8b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8b.id}, selecting best answer(s) and mapping to options, thinking: {thinking8b.content}; answer: {answer8b.content}")
    sub_tasks.append(f"Sub-task 8b output: thinking - {thinking8b.content}; answer - {answer8b.content}")
    subtask_desc_8b['response'] = {"thinking": thinking8b, "answer": answer8b}
    logs.append(subtask_desc_8b)
    print("Step 8b: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8b, answer8b, sub_tasks, agents)
    return final_answer, logs
