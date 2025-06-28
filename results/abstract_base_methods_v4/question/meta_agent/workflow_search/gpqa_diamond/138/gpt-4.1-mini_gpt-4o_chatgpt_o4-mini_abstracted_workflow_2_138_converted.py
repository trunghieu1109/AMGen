async def forward_138(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    stage1_subtask1_instruction = (
        "Sub-task 1: Research and explicitly identify the detailed chemical mechanism of the reaction between compounds A and B with NaNO2, HCl, and H2O. "
        "Emphasize that the reaction involves nitrosation, diazotization, and possible rearrangements rather than direct oxidation to diketones. "
        "Summarize key mechanistic pathways relevant to the substrates, including nitrosation of amines, diazonium salt formation, and oxidative cleavage of vicinal diols. "
        "Include bullet-point summaries of known HNO2 reactions to guide reasoning."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": stage1_subtask1_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], stage1_subtask1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying reaction mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    stage1_subtask1_debate_instruction = (
        "Sub-task 1 Debate: Critically evaluate and debate the identified reaction mechanism from Sub-task 1. "
        "Challenge assumptions and confirm the involvement of nitrosation, diazotization, and rearrangements rather than oxidation. "
        "Use evidence and chemical knowledge to validate or revise the mechanism summary."
    )
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking1 = [[] for _ in range(N_max_1)]
    all_answer1 = [[] for _ in range(N_max_1)]
    subtask_desc1_debate = {
        "subtask_id": "subtask_1_debate",
        "instruction": stage1_subtask1_debate_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking_d1, answer_d1 = await agent([taskInfo, thinking1, answer1], stage1_subtask1_debate_instruction, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking1, answer1] + all_thinking1[r-1] + all_answer1[r-1]
                thinking_d1, answer_d1 = await agent(input_infos_1, stage1_subtask1_debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating reaction mechanism, thinking: {thinking_d1.content}; answer: {answer_d1.content}")
            all_thinking1[r].append(thinking_d1)
            all_answer1[r].append(answer_d1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_final, answer1_final = await final_decision_agent_1([taskInfo] + all_thinking1[-1] + all_answer1[-1], "Sub-task 1: Make final decision on the confirmed reaction mechanism.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing reaction mechanism, thinking: {thinking1_final.content}; answer: {answer1_final.content}")
    sub_tasks.append(f"Sub-task 1 debate output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    subtask_desc1_debate["response"] = {"thinking": thinking1_final, "answer": answer1_final}
    logs.append(subtask_desc1_debate)
    print("Step 1 Debate: ", sub_tasks[-1])

    stage1_subtask2_instruction = (
        "Sub-task 2: Based on the confirmed reaction mechanism from Sub-task 1, predict plausible chemical transformations and intermediates that could lead to the formation of the diketones "
        "4-isopropylcyclohexane-1,2-dione and 5-methylhexane-2,3-dione from appropriate precursors. Include possible rearrangements or oxidative cleavage steps consistent with nitrous acid chemistry. "
        "Generate multiple hypotheses and select the most chemically plausible one using self-consistency."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": stage1_subtask2_instruction,
        "context": ["user query", "thinking of subtask 1 debate", "answer of subtask 1 debate"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1_final, answer1_final], stage1_subtask2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, predicting chemical transformations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    stage1_subtask3_instruction = (
        "Sub-task 3: Analyze the functional groups and structural features of each given starting material choice (A and B) to verify their compatibility with the reaction mechanism and their potential to yield the specified diketones under the reaction conditions. "
        "Explicitly check for presence of functional groups such as α-amino ketones, vicinal diols, or other groups known to undergo transformations with NaNO2/HCl/H2O. "
        "Use reflexion with critic feedback to iteratively refine the evaluation."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1_final, answer1_final, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": stage1_subtask3_instruction,
        "context": ["user query", "thinking of subtask 1 debate", "answer of subtask 1 debate", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, stage1_subtask3_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing functional groups and compatibility, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Please review the functional group analysis and compatibility evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, stage1_subtask3_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining functional group evaluation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    stage2_subtask4_instruction = (
        "Sub-task 4: Critically evaluate and debate the plausibility of each choice (1 to 4) as the correct pair of starting materials A and B, integrating mechanistic insights, predicted transformations, "
        "and functional group compatibility from previous subtasks. Use a self-consistency and debate approach to select the most chemically reasonable choice that leads to the formation of the target diketones."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": stage2_subtask4_instruction,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], stage2_subtask4_instruction, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, stage2_subtask4_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating correct starting materials, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the correct starting materials A and B.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    stage2_subtask5_instruction = (
        "Sub-task 5: Provide the final answer by selecting the correct multiple-choice option (A, B, C, or D) corresponding to the starting materials A and B that produce the diketones 4-isopropylcyclohexane-1,2-dione and 5-methylhexane-2,3-dione respectively, "
        "based on the comprehensive evaluation in Sub-task 4."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": stage2_subtask5_instruction,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], stage2_subtask5_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, providing final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
