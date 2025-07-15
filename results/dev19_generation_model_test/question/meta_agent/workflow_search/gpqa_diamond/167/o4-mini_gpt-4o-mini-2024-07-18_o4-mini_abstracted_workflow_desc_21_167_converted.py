async def forward_167(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1, Sub-task 1: SC_CoT
    cot_sc_instruction = (
        "Sub-task 1: Extract and clarify the question context and key definitions, "
        "explicitly define 'difficult-to-spot' (silent) errors vs obvious failures, "
        "and agree on the meaning of 'most common'."
    )
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent definitions and context.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 1, Sub-task 2: SC_CoT
    cot_sc_instruction2 = (
        "Sub-task 2: Standardize and document precise scopes and examples for each listed issue: "
        "1. Mutually incompatible data formats; 2. 'chr'/ 'no chr' confusion; 3. Reference assembly mismatch; 4. Incorrect ID conversion. "
        "Clarify which parts could produce subtle errors vs outright failures."
    )
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent scopes and examples.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 2, Sub-task 3: Debate
    debate_instruction3 = (
        "Sub-task 3: Evaluate each of the four issues against criteria—frequency, subtlety (with at least one example of silent error vs obvious failure), downstream impact. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinkings3 = []
    all_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction3,"context":["thinking of subtask_2","answer of subtask_2"],"agent_collaboration":"Debate"}
    for agent in debate_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction3, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        all_thinkings3.append(thinking3_i)
        all_answers3.append(answer3_i)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + all_thinkings3 + all_answers3,
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final evaluation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 2, Sub-task 4: Reflexion
    reflect_inst4 = ("Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
                     "Using insights from previous attempts, try to solve the task better.")
    cot_reflect_instruction4 = "Sub-task 4: Conduct a critical review of subtask_3’s evaluations to challenge any issue that does not yield difficult-to-spot errors. " + reflect_inst4
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction4,
                     "context":["thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content.strip() == "True":
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4, "answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Stage 3, Sub-task 5: Debate
    debate_instruction5 = (
        "Sub-task 5: Aggregate the surviving issues from subtask_4 based on their scores (frequency + subtlety), map to answer choices and rank them. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinkings5 = []
    all_answers5 = []
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":debate_instruction5,
                     "context":["thinking of subtask_4","answer of subtask_4"],"agent_collaboration":"Debate"}
    for agent in debate_agents5:
        thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        all_thinkings5.append(thinking5_i)
        all_answers5.append(answer5_i)
    final_decision5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo, thinking4, answer4] + all_thinkings5 + all_answers5,
        "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final ranking and mapping to choices.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5, "answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    # Stage 3, Sub-task 6: SC_CoT
    cot_sc_instruction6 = (
        "Sub-task 6: Select the correct answer choice and craft a concise justification citing the scoring rubric, examples of silent errors, "
        "and reasons for exclusion of other issues."
    )
    N6 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_thinkings6 = []
    possible_answers6 = []
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_sc_instruction6,
                     "context":["thinking of subtask_5","answer of subtask_5"],"agent_collaboration":"SC_CoT"}
    for i in range(N6):
        thinking6_i, answer6_i = await cot_agents6[i]([taskInfo, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_thinkings6.append(thinking6_i)
        possible_answers6.append(answer6_i)
    final_decision6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision6([taskInfo, thinking5, answer5] + possible_thinkings6 + possible_answers6,
        "Sub-task 6: Synthesize and choose the most consistent final answer and justification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking":thinking6, "answer":answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs