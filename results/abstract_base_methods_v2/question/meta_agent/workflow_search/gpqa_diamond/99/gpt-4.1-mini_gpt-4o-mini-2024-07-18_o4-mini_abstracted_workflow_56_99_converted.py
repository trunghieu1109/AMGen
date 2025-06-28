async def forward_99(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify the possible isomers of compound A (C3H6), determine the most plausible structure of A, and predict the product B formed after bromination in carbon tetrachloride, including mechanistic rationale."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying compound A and product B, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Determine the structure and properties of compound C formed when compound B reacts with alcoholic KOH, including the elimination mechanism and physical state (e.g., gas, liquid), based on Sub-task 1 outputs."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining compound C, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3a = "Sub-task 3a: List all plausible products formed when compound C is passed through a red-hot iron tube, considering typical pyrolysis/dehydrohalogenation reactions of C3 hydrocarbons and their isomers, based on Sub-task 2 outputs."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, listing plausible products of compound C pyrolysis, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_instruction_3b = "Sub-task 3b: Evaluate the chemical plausibility of each product from Sub-task 3a using standard organic chemistry knowledge and select the most reasonable compound D, providing mechanistic and structural justification."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, evaluating and selecting compound D, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Determine the structure and characteristics of compound E formed when compound D reacts with a mixture of two strong acids, considering possible addition or substitution reactions and their products, based on Sub-task 3b outputs."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3b, answer3b], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining compound E, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Identify compound F formed when compound E reacts with iron scrap and hydrochloric acid, analyze its structure, and assess its known applications such as dye synthesis, based on Sub-task 4 outputs."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying compound F and evaluating dye use, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on compound F and its applications.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on compound F and dye synthesis, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_sc_instruction_6 = "Sub-task 6: Determine the structure and properties of compound G formed when compound F reacts with nitrous acid, including the reaction mechanism and expected functional groups, based on Sub-task 5 outputs."
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, determining compound G, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    most_common_answer_6 = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[most_common_answer_6]
    answer6 = answermapping_6[most_common_answer_6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Identify compound H formed when compound G reacts with sodium hydroxide, analyze its chemical behavior, including expected color reactions with ferric chloride solution, based on Sub-task 6 outputs."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, identifying compound H and analyzing chemical behavior, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Perform a reflexion and self-consistency check on the assigned structures of compounds D, E, F, G, and H to verify chemical plausibility and consistency with known reaction pathways and spectral data, based on Sub-tasks 3b, 4, 5, 6, and 7 outputs."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking3b, answer3b, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking and answer of subtask 3b", "thinking and answer of subtask 4", "thinking and answer of subtask 5", "thinking and answer of subtask 6", "thinking and answer of subtask 7"],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, checking chemical plausibility and consistency, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8], "please review the chemical plausibility and consistency check and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining plausibility and consistency check, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    cot_sc_instruction_9 = "Sub-task 9: Analyze the 1H NMR spectral data of compound D to verify if it gives two singlets, based on the confirmed structure from Sub-task 3b and consistency check from Sub-task 8."
    cot_agents_9 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_9 = []
    thinkingmapping_9 = {}
    answermapping_9 = {}
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_sc_instruction_9,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b", "thinking and answer of subtask 8"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking9, answer9 = await cot_agents_9[i]([taskInfo, thinking3b, answer3b, thinking8, answer8], cot_sc_instruction_9, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_9[i].id}, analyzing NMR of compound D, thinking: {thinking9.content}; answer: {answer9.content}")
        possible_answers_9.append(answer9.content)
        thinkingmapping_9[answer9.content] = thinking9
        answermapping_9[answer9.content] = answer9
    most_common_answer_9 = Counter(possible_answers_9).most_common(1)[0][0]
    thinking9 = thinkingmapping_9[most_common_answer_9]
    answer9 = answermapping_9[most_common_answer_9]
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    cot_sc_instruction_10 = "Sub-task 10: Evaluate the validity of the statement that compound F is used for the synthesis of dyes, based on its confirmed structure and known industrial applications from Sub-task 5 and consistency check from Sub-task 8."
    cot_agents_10 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_10 = []
    thinkingmapping_10 = {}
    answermapping_10 = {}
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_sc_instruction_10,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking and answer of subtask 8"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking10, answer10 = await cot_agents_10[i]([taskInfo, thinking5, answer5, thinking8, answer8], cot_sc_instruction_10, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_10[i].id}, evaluating dye synthesis use of compound F, thinking: {thinking10.content}; answer: {answer10.content}")
        possible_answers_10.append(answer10.content)
        thinkingmapping_10[answer10.content] = thinking10
        answermapping_10[answer10.content] = answer10
    most_common_answer_10 = Counter(possible_answers_10).most_common(1)[0][0]
    thinking10 = thinkingmapping_10[most_common_answer_10]
    answer10 = answermapping_10[most_common_answer_10]
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    cot_sc_instruction_11 = "Sub-task 11: Assess whether compound H gives a yellow color with the addition of ferric chloride solution, based on its functional groups and chemical properties confirmed in Sub-task 7 and consistency check from Sub-task 8."
    cot_agents_11 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_11 = []
    thinkingmapping_11 = {}
    answermapping_11 = {}
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_sc_instruction_11,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7", "thinking and answer of subtask 8"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking11, answer11 = await cot_agents_11[i]([taskInfo, thinking7, answer7, thinking8, answer8], cot_sc_instruction_11, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_11[i].id}, assessing color reaction of compound H, thinking: {thinking11.content}; answer: {answer11.content}")
        possible_answers_11.append(answer11.content)
        thinkingmapping_11[answer11.content] = thinking11
        answermapping_11[answer11.content] = answer11
    most_common_answer_11 = Counter(possible_answers_11).most_common(1)[0][0]
    thinking11 = thinkingmapping_11[most_common_answer_11]
    answer11 = answermapping_11[most_common_answer_11]
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])
    
    cot_sc_instruction_12 = "Sub-task 12: Determine if compound C is a flammable gas by analyzing its structure and physical properties established in Sub-task 2."
    cot_agents_12 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_12 = []
    thinkingmapping_12 = {}
    answermapping_12 = {}
    subtask_desc12 = {
        "subtask_id": "subtask_12",
        "instruction": cot_sc_instruction_12,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking12, answer12 = await cot_agents_12[i]([taskInfo, thinking2, answer2], cot_sc_instruction_12, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_12[i].id}, determining flammability of compound C, thinking: {thinking12.content}; answer: {answer12.content}")
        possible_answers_12.append(answer12.content)
        thinkingmapping_12[answer12.content] = thinking12
        answermapping_12[answer12.content] = answer12
    most_common_answer_12 = Counter(possible_answers_12).most_common(1)[0][0]
    thinking12 = thinkingmapping_12[most_common_answer_12]
    answer12 = answermapping_12[most_common_answer_12]
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    subtask_desc12['response'] = {"thinking": thinking12, "answer": answer12}
    logs.append(subtask_desc12)
    print("Step 12: ", sub_tasks[-1])
    
    cot_reflect_instruction_13 = "Sub-task 13: Compare and cross-validate the four given statements (choices) against the analyzed data of compounds D, F, H, and C to identify the incorrect statement, based on Sub-tasks 9, 10, 11, and 12 outputs."
    cot_agent_13 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_13 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_13 = self.max_round
    cot_inputs_13 = [taskInfo, thinking9, answer9, thinking10, answer10, thinking11, answer11, thinking12, answer12]
    subtask_desc13 = {
        "subtask_id": "subtask_13",
        "instruction": cot_reflect_instruction_13,
        "context": ["user query", "thinking and answer of subtask 9", "thinking and answer of subtask 10", "thinking and answer of subtask 11", "thinking and answer of subtask 12"],
        "agent_collaboration": "Reflexion"
    }
    thinking13, answer13 = await cot_agent_13(cot_inputs_13, cot_reflect_instruction_13, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_13.id}, identifying incorrect statement, thinking: {thinking13.content}; answer: {answer13.content}")
    for i in range(N_max_13):
        feedback, correct = await critic_agent_13([taskInfo, thinking13, answer13], "please review the identification of the incorrect statement and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_13.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_13.extend([thinking13, answer13, feedback])
        thinking13, answer13 = await cot_agent_13(cot_inputs_13, cot_reflect_instruction_13, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_13.id}, refining incorrect statement identification, thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    subtask_desc13['response'] = {"thinking": thinking13, "answer": answer13}
    logs.append(subtask_desc13)
    print("Step 13: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking13, answer13, sub_tasks, agents)
    return final_answer, logs
