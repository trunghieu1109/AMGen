async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_sc = self.max_sc
    N_round = self.max_round
    subtask_1_instruction = "Sub-task 1: Analyze the structure and functional groups of the starting material ((2,2-dimethylbut-3-en-1-yl)oxy)benzene to identify all reactive sites, including the alkene moiety and the ether oxygen, and assess their potential roles in the reaction with hydrogen bromide."
    subtask_1_desc = {
        "subtask_id": "subtask_1",
        "instruction": subtask_1_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], subtask_1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting material structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_1_desc["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_1_desc)
    print("Step 1: ", sub_tasks[-1])
    subtask_2_instruction = "Sub-task 2: Review the chemical behavior and reaction mechanisms of hydrogen bromide (HBr) with alkenes and ethers, explicitly including electrophilic addition to alkenes, cleavage of ether bonds, and intramolecular cyclisation involving the ether oxygen attacking carbocation intermediates."
    subtask_2_desc = {
        "subtask_id": "subtask_2",
        "instruction": subtask_2_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinking_map_2 = {}
    answer_map_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], subtask_2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, reviewing HBr behavior including electrophilic addition, ether cleavage, and intramolecular cyclisation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinking_map_2[answer2.content] = thinking2
        answer_map_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinking_map_2[answer2_content]
    answer2 = answer_map_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_2_desc["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_2_desc)
    print("Step 2: ", sub_tasks[-1])
    subtask_3_instruction = "Sub-task 3: Generate and describe all plausible reaction pathways between the starting material and HBr, covering electrophilic addition to the alkene, ether cleavage, and intramolecular cyclisation mechanisms, considering carbocation stability and reaction conditions that favor cyclisation."
    subtask_3_desc = {
        "subtask_id": "subtask_3",
        "instruction": subtask_3_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, subtask_3_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, generating plausible reaction pathways, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_7([taskInfo, thinking3, answer3], "please review the generated reaction pathways and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, subtask_3_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining reaction pathways, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_3_desc["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_3_desc)
    print("Step 3: ", sub_tasks[-1])
    subtask_4_instruction = "Sub-task 4: Enumerate, draw, and characterize all plausible product structures arising from the predicted reaction pathways, including linear addition products, ether cleavage products, and cyclic products such as chromane and dihydrobenzofuran derivatives, with attention to regioselectivity and stereochemistry."
    subtask_4_desc = {
        "subtask_id": "subtask_4",
        "instruction": subtask_4_instruction,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], subtask_4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, enumerating plausible products including cyclic and linear products, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_4_desc["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_4_desc)
    print("Step 4: ", sub_tasks[-1])
    subtask_5_instruction = "Sub-task 5: Compare the enumerated plausible product structures with the given multiple-choice options, mapping each product or set of products to the corresponding choices, and noting which options include cyclic products consistent with intramolecular cyclisation."
    subtask_5_desc = {
        "subtask_id": "subtask_5",
        "instruction": subtask_5_instruction,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    all_thinking5 = [[] for _ in range(N_round)]
    all_answer5 = [[] for _ in range(N_round)]
    for r in range(N_round):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], subtask_5_instruction, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, subtask_5_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing products with options, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Map plausible products to multiple-choice options, allowing multiple selections if justified.", is_sub_task=True)
    agents.append(f"Final Decision agent, mapping products to options, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_5_desc["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_5_desc)
    print("Step 5: ", sub_tasks[-1])
    subtask_6_instruction = "Sub-task 6: Interpret the TLC data indicating the disappearance of the reactant spot and the appearance of two new spots, correlating this observation with the number and nature of predicted products and their representation in the multiple-choice options."
    subtask_6_desc = {
        "subtask_id": "subtask_6",
        "instruction": subtask_6_instruction,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], subtask_6_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, interpreting TLC results and correlating with predicted products, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_6_desc["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_6_desc)
    print("Step 6: ", sub_tasks[-1])
    subtask_7_instruction = "Sub-task 7: Conduct a combined debate and reflexion to evaluate all plausible products and their consistency with the reaction mechanism and TLC data, allowing for multiple products and multiple-choice answers if chemically justified, and finalize the selection of the correct answer(s) accordingly."
    subtask_7_desc = {
        "subtask_id": "subtask_7",
        "instruction": subtask_7_instruction,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Debate+Reflexion"
    }
    all_thinking7 = [[] for _ in range(N_round)]
    all_answer7 = [[] for _ in range(N_round)]
    for r in range(N_round):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], subtask_7_instruction, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, subtask_7_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating final answer options, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Finalize the selection of correct multiple-choice answer(s) based on all evidence and reasoning, allowing multiple answers if justified.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer selection, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_7_desc["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_7_desc)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs